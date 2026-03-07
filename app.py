from flask import Flask, request, jsonify, render_template, session
from groq import Groq
from model import score_transaction
from dotenv import load_dotenv
import os
import base64

load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)

groq_client = Groq(api_key=os.getenv('GROQ_API_KEY'))

SYSTEM_PROMPT = """You are FraudGuard, an intelligent fraud detection assistant powered by a real XGBoost machine learning model trained on 590,540 real financial transactions.

YOUR PERSONALITY:
- Conversational, confident, and clear. Talk like a knowledgeable friend, not a robot.
- Match response length to the question. Greetings get short replies. Fraud analysis gets full explanations. Education questions get thorough answers.
- Never use unnecessary filler. Every sentence should add value.

STRICT RULES FOR TRANSACTION ANALYSIS:
- The ML model verdict is FINAL. Never change the probability, never contradict it.
- Always explain in plain English what caused the fraud — what specific pattern made the model suspicious.
- Classify the fraud type (CNP, ATO, Card Testing) and explain what that means in context.
- Tell the user exactly what to do right now.

RESPONSE FORMAT FOR TRANSACTIONS:
Start with the verdict clearly. Then explain what caused it as if talking to someone who knows nothing about fraud. Then classify the fraud type. Then tell them what to do. Use natural paragraphs, not bullet points.

FOR GENERAL QUESTIONS:
Answer thoroughly. If someone asks how a fraud type works, explain it properly with real examples. If someone asks what to do after fraud, walk them through it step by step.

FOR PREVENTION:
Give specific, practical advice tied to the situation. Not generic tips.

WHEN GREETED:
Introduce yourself warmly in 2-3 sentences. Tell them what you can help with and give one example to get them started."""
MODE_PROMPTS = {
    'transaction': "The user wants to analyse a specific transaction. Extract details and score it with the ML model.",
    'education': "The user wants to learn about fraud. Answer their question in 3 sentences max. No model scoring needed.",
    'prevention': "The user wants prevention advice. Give exactly 3 specific actionable steps. No model scoring needed.",
    'report': "The user wants a structured risk report. Use this format exactly:\nRISK LEVEL: [level]\nFRAUD TYPE: [type]\nPROBABILITY: [%]\nTOP FLAGS: [list]\nRECOMMENDED ACTION: [one sentence]"
}

def extract_transaction_from_text(user_message, conversation_history):
    """Use Groq to extract transaction details from natural language."""
    
    extraction_prompt = """Extract transaction details from the user message and return ONLY a JSON object.

If an amount is mentioned in a non-USD currency, convert it to USD using these approximate rates:
- INR (rupees): divide by 83
- GBP (pounds): multiply by 1.27
- EUR (euros): multiply by 1.08
- AED (dirhams): divide by 3.67
- CAD: multiply by 0.74
- AUD: multiply by 0.65
- SGD: multiply by 0.74
If currency is unclear, assume USD.

Return this exact format with no extra text:
{
  "amount": number in USD or null,
  "hour": number 0-23 or null,
  "is_mobile": 1 or 0 or null,
  "is_free_email": 1 or 0 or null,
  "email_match": 1 or 0 or null,
  "has_identity": 1 or 0 or null,
  "original_currency": "USD" or currency code,
  "original_amount": original number or null,
  "is_asking_question": true or false,
  "has_transaction": true or false
}"""

    response = groq_client.chat.completions.create(
       model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {"role": "system", "content": extraction_prompt},
            {"role": "user", "content": user_message}
        ],
        temperature=0
    )
    
    import json
    try:
        text = response.choices[0].message.content.strip()
        # clean up markdown if present
        if '```' in text:
            text = text.split('```')[1]
            if text.startswith('json'):
                text = text[4:]
        return json.loads(text.strip())
    except:
        return {"has_transaction": False, "is_asking_question": True}

@app.route('/')
def index():
    session['conversation'] = []
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '')
    image_data = data.get('image', None)
    
    if 'conversation' not in session:
        session['conversation'] = []
    
    conversation = session['conversation']
    
    # if image uploaded, extract transaction details from it
    if image_data:
        image_content = [
            {
                "type": "image_url",
                "image_url": {"url": image_data}
            },
            {
                "type": "text", 
                "text": "Extract all transaction details from this payment screenshot. What is the amount, time, merchant, and any other relevant details?"
            }
        ]
        
        image_response = groq_client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": image_content}]
        )
        user_message = image_response.choices[0].message.content
    
    # extract transaction details
    extracted = extract_transaction_from_text(user_message, conversation)
    
    # build context for groq
    mode = data.get('mode', 'transaction')
    mode_context = MODE_PROMPTS.get(mode, MODE_PROMPTS['transaction'])
    messages = [{"role": "system", "content": SYSTEM_PROMPT + "\n\nCurrent mode: " + mode_context}]
    messages.extend(conversation[-10:])  # last 10 messages for memory
    
    if extracted.get('has_transaction'):
        # score with ml model
        transaction_info = {k: v for k, v in extracted.items() 
                          if k not in ['is_asking_question', 'has_transaction'] and v is not None}
        
        result = score_transaction(transaction_info)
        
        # tell groq the model results
        analysis_context = f"""The ML model has scored this transaction. These results are FINAL and you must not change them:

Original Amount: {extracted.get('original_amount')} {extracted.get('original_currency')} (converted to ${transaction_info.get('amount')} USD for model scoring)
Fraud Probability: {result['probability']}%
Verdict: {result['verdict']}
Risk Level: {result['risk_level']}
Flags: {', '.join(result['flags']) if result['flags'] else 'none'}

User message: {user_message}

Follow the exact response format. Do not change the verdict or probability."""

        messages.append({"role": "user", "content": analysis_context})
    else:
        messages.append({"role": "user", "content": user_message})
    
    # get groq response
    response = groq_client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=messages,
        temperature=0.7,
        max_tokens=1000
    )
    
    assistant_message = response.choices[0].message.content
    
    # update conversation memory
    conversation.append({"role": "user", "content": user_message})
    conversation.append({"role": "assistant", "content": assistant_message})
    session['conversation'] = conversation
    session.modified = True
    
    result_data = {'message': assistant_message}
    if extracted.get('has_transaction'):
        result_data['fraud_data'] = result
    
    return jsonify(result_data)

@app.route('/clear', methods=['POST'])
def clear():
    session['conversation'] = []
    return jsonify({'status': 'cleared'})

if __name__ == '__main__':
    app.run(debug=True)