from flask import Flask, request, jsonify, render_template, session
from groq import Groq
from model import score_transaction
from dotenv import load_dotenv
import os
import json

load_dotenv(override=True)

app = Flask(__name__)
app.secret_key = os.urandom(24)

groq_client = Groq(api_key=os.getenv('GROQ_API_KEY'))

SYSTEM_PROMPT = """You are FraudGuard — a fraud intelligence system backed by a real machine learning model trained on 1,000,000 bank account transactions from the NeurIPS 2022 Bank Account Fraud Dataset.

You think like a seasoned fraud analyst who also happens to be great at explaining things. You've seen thousands of fraud cases. You know the patterns. When someone describes a transaction or an application, you don't just run it through a checklist — you actually think about what's going on, what the likely story is, and what the person should do.

You talk like a person. Not an assistant, not a bot, not a product. A person who knows a lot about fraud and genuinely wants to help. You're direct, you're clear, and you don't waste words. When something is suspicious you say so plainly. When something looks fine you say that too without hedging.

Always respond in English by default. Only switch to another language if the user explicitly writes to you in that language first.

You remember everything said in the conversation. If someone mentioned a transaction earlier and now asks a follow up, you connect the dots without being asked.

When someone describes a transaction or suspicious activity, think through it naturally — what patterns stand out, what type of fraud does this look like. Then score it using your ML model and explain what the model found in plain English. Not jargon, not a template — actual explanation of why this specific situation looks suspicious or not.

ONLY output a <<SCORE>> block when the user is clearly describing a specific suspicious transaction or financial activity they want analysed. Never output a <<SCORE>> block for greetings, general questions, or casual conversation.

To score a transaction, extract what you can from what the user said and output this block at the very END of your response, after everything else:
<<SCORE>>
{"housing_status": 0, "device_os": 2, "has_other_cards": 0, "keep_alive_session": 0, "phone_home_valid": 0, "email_is_free": 1, "income": 0.3, "prev_address_months_count": 2, "foreign_request": 1, "credit_risk_score": 80}
<</SCORE>>

Only include fields you are confident about from what the user described. Use these mappings:
- renting / no fixed address / unstable housing = housing_status: 0, owns home = housing_status: 3, average housing = housing_status: 2
- Windows = device_os: 0, Mac = device_os: 1, Linux = device_os: 2, Android = device_os: 3, iOS = device_os: 4
- has no other bank cards = has_other_cards: 0, has other cards = has_other_cards: 1
- session ended quickly or no keep alive = keep_alive_session: 0, normal session = keep_alive_session: 1
- no home phone or invalid = phone_home_valid: 0, has valid home phone = phone_home_valid: 1
- gmail / yahoo / hotmail / free email = email_is_free: 1, work or corporate email = email_is_free: 0
- income is a decimal 0 to 1 (0.1 = very low, 0.5 = average, 0.9 = high)
- prev_address_months_count: months at previous address, 0 if just moved or unknown
- foreign_request: 1 if request is coming from another country, 0 if local
- credit_risk_score: 0 to 300, higher is better (below 100 = risky, 150 = average, 250+ = good)

The ML model's probability is final. You explain it, you never argue with it or change it.

When someone just says hi or starts casual conversation, respond naturally in one or two sentences. Never list your features or introduce yourself with a paragraph.

When someone asks a general question about fraud — how phishing works, what to do if their card is stolen, how to spot a scam — answer it properly with real knowledge. Give them something genuinely useful, not generic advice they could have googled.

For prevention advice, be specific to their situation. Not generic tips.

You are not a demo. You are not a prototype. You are a working fraud intelligence system and you act like one."""


def extract_score_block(text):
    if '<<SCORE>>' not in text:
        return text, None
    try:
        start = text.index('<<SCORE>>') + len('<<SCORE>>')
        end = text.index('<</SCORE>>') if '<</SCORE>>' in text else len(text)
        clean_text = text[:text.index('<<SCORE>>')].strip()
        json_str = text[start:end].strip()
        transaction_info = json.loads(json_str)
        return clean_text, transaction_info
    except Exception as e:
        print(f"Score block parse error: {e}")
        return text, None


@app.route('/')
def index():
    session['conversation'] = []
    return render_template('index.html')


@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'model': 'fraudguard_v2', 'version': '2.1'}), 200


@app.route('/batch')
def batch():
    return render_template('batch.html')


@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json or {}
        user_message = data.get('message', '')
        image_data = data.get('image', None)
        mode = data.get('mode', 'transaction')

        if 'conversation' not in session:
            session['conversation'] = []

        conversation = session['conversation']

        if image_data:
            try:
                image_response = groq_client.chat.completions.create(
                    model="meta-llama/llama-4-scout-17b-16e-instruct",
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": image_data}},
                            {"type": "text", "text": "Extract all transaction details from this payment screenshot — amount, currency, time, merchant, device type, email addresses if visible, and anything else relevant to fraud detection."}
                        ]
                    }]
                )
                user_message = image_response.choices[0].message.content
            except Exception as e:
                print(f"Vision error: {e}")
                user_message = "I uploaded a payment screenshot but couldn't read it clearly. Can you help me assess if it might be fraud?"

        mode_hints = {
            'education': ' (user wants to learn about fraud)',
            'prevention': ' (user wants prevention advice)',
            'report': ' (user wants a structured risk report)'
        }

        message_with_hint = user_message
        if mode != 'transaction' and mode in mode_hints:
            message_with_hint = user_message + mode_hints[mode]

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(conversation[-12:])
        messages.append({"role": "user", "content": message_with_hint})

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.7,
            max_tokens=800
        )

        raw_response = response.choices[0].message.content
        clean_response, transaction_info = extract_score_block(raw_response)

        transaction_keywords = ['transaction', 'payment', 'transfer', 'charge', 'purchase',
                                'fraud', 'suspicious', 'account', 'card', 'bank', 'money',
                                'amount', 'sent', 'received', 'debit', 'credit', 'upi',
                                'withdraw', 'deposit', 'loan', 'request', 'email', 'device']
        message_lower = (user_message or '').lower()
        has_transaction_context = any(word in message_lower for word in transaction_keywords)

        if not has_transaction_context:
            transaction_info = None

        fraud_result = None
        if transaction_info:
            try:
                fraud_result = score_transaction(transaction_info)
                verdict_emoji = "🔴" if fraud_result['verdict'] == 'FRAUD' else "🟢"
                score_summary = f"\n\n{verdict_emoji} Risk Flag: {fraud_result['risk_level']} ({fraud_result['probability']}% model confidence)"
                clean_response = clean_response + score_summary
            except Exception as e:
                print(f"Scoring error: {e}")

        conversation.append({"role": "user", "content": user_message})
        conversation.append({"role": "assistant", "content": clean_response})
        session['conversation'] = conversation
        session.modified = True

        result = {'message': clean_response}
        if fraud_result:
            result['fraud_data'] = fraud_result

        return jsonify(result)

    except Exception as e:
        print(f"Chat endpoint error: {e}")
        return jsonify({'message': 'Something went wrong. Please try again.', 'fraud_data': None}), 200


@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Direct transaction scoring endpoint — no LLM, just the model.
    Used for API integrations and testing.

    POST /analyze
    {
        "housing_status": 0,
        "email_is_free": 1,
        "foreign_request": 1,
        "credit_risk_score": 80,
        "amount": 75000,
        "hour": 2
    }
    """
    try:
        transaction_info = request.json or {}
        result = score_transaction(transaction_info)
        return jsonify({
            'status': 'success',
            'risk_score': result['hybrid_score'],
            'risk_level': result['risk_level'],
            'verdict': result['verdict'],
            'probability': result['probability'],
            'signals': result['signals'],
            'flags': result['flags'],
            'explanation': result['explanation'],
            'actions': result['actions'],
            'threshold_used': result['threshold_used']
        }), 200
    except Exception as e:
        print(f"Analyze endpoint error: {e}")
        return jsonify({
            'status': 'error',
            'risk_score': 0,
            'risk_level': 'Low — Clear',
            'verdict': 'LEGITIMATE',
            'probability': 0,
            'signals': [],
            'flags': [],
            'explanation': 'Scoring unavailable. Please try again.',
            'actions': ['allow transaction'],
            'threshold_used': 10.0
        }), 200


@app.route('/analyze/batch', methods=['POST'])
def analyze_batch():
    """
    Batch transaction scoring — processes multiple transactions at once.
    Designed for bank/fintech use cases handling high-volume transaction feeds.

    POST /analyze/batch
    {
        "transactions": [
            {"id": "txn_001", "amount": 75000, "hour": 2, "foreign_request": 1},
            {"id": "txn_002", "amount": 500, "email_is_free": 0},
            ...
        ]
    }

    Returns individual results + summary breakdown.
    """
    try:
        body = request.json or {}
        transactions = body.get('transactions', [])

        if not isinstance(transactions, list):
            return jsonify({'status': 'error', 'message': 'transactions must be a list'}), 400

        if len(transactions) > 500:
            return jsonify({'status': 'error', 'message': 'batch limit is 500 transactions'}), 400

        results = []
        summary = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0, 'total': len(transactions)}

        for txn in transactions:
            if not isinstance(txn, dict):
                continue

            txn_id = txn.get('id', f"txn_{len(results) + 1}")

            try:
                scored = score_transaction(txn)
                risk_label = scored['risk_level'].split('—')[0].strip().lower()

                if risk_label in summary:
                    summary[risk_label] += 1
                else:
                    summary['low'] += 1

                results.append({
                    'id': txn_id,
                    'status': 'scored',
                    'risk_score': scored['hybrid_score'],
                    'risk_level': scored['risk_level'],
                    'verdict': scored['verdict'],
                    'probability': scored['probability'],
                    'signals': scored['signals'],
                    'flags': scored['flags'],
                    'explanation': scored['explanation'],
                    'actions': scored['actions']
                })
            except Exception as e:
                print(f"Batch item error for {txn_id}: {e}")
                results.append({
                    'id': txn_id,
                    'status': 'error',
                    'risk_score': 0,
                    'risk_level': 'Low — Clear',
                    'verdict': 'LEGITIMATE',
                    'probability': 0,
                    'signals': [],
                    'flags': [],
                    'explanation': 'Scoring failed for this transaction.',
                    'actions': ['allow transaction']
                })
                summary['low'] += 1

        fraud_count = sum(1 for r in results if r.get('verdict') == 'FRAUD')

        return jsonify({
            'status': 'success',
            'processed': len(results),
            'summary': {
                'critical': summary['critical'],
                'high': summary['high'],
                'medium': summary['medium'],
                'low': summary['low'],
                'total': summary['total'],
                'flagged': fraud_count
            },
            'results': results
        }), 200

    except Exception as e:
        print(f"Batch endpoint error: {e}")
        return jsonify({
            'status': 'error',
            'message': 'Batch processing failed. Please try again.',
            'processed': 0,
            'summary': {'critical': 0, 'high': 0, 'medium': 0, 'low': 0, 'total': 0, 'flagged': 0},
            'results': []
        }), 200


@app.route('/clear', methods=['POST'])
def clear():
    session['conversation'] = []
    return jsonify({'status': 'cleared'})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)