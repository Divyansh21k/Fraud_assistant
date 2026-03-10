# FraudGuard — Fraud Detection Assistant

A conversational fraud detection assistant powered by a real machine learning model. Describe any transaction in plain English and get an instant fraud risk assessment.

Live: https://fraud-assistant.onrender.com | 

## What is this?

After completing Phase 1 of my fraud detection project on the IEEE-CIS Kaggle competition, I realized the model I built there could not be used in a real product. The features were all proprietary Vesta columns with names like C13 and V258. No real person knows what those mean.

So I retrained on the NeurIPS 2022 Bank Account Fraud Dataset which has fully human readable features like housing status, device type, email provider, and credit score. Then I built FraudGuard around it. You just describe what happened and the model scores it.

## Project Structure
```
Fraud_assistant/
├── app.py                  # Flask backend, Groq API integration, chat routes
├── model.py                # Loads ML model, scores transactions, returns risk verdict
├── fraudguard_v2.pkl       # Trained XGBoost model (Git LFS)
├── requirements.txt        # Python dependencies
├── templates/
│   └── index.html          # Bloomberg terminal UI, History and Reports panels
└── static/
    └── style.css           # Additional styles
```

## How it works

You type a description of a transaction or suspicious activity. The LLM reads your message, extracts the relevant features like whether you used a free email, whether the request came from a foreign country, what your housing situation is, and maps them to the model's input format. The XGBoost model then scores the transaction and returns a fraud probability. The LLM explains what the model found in plain English.

The model flags transactions for review rather than making a definitive verdict. This is intentional and reflects how real fraud systems work.

## The ML Model

Dataset: NeurIPS 2022 Bank Account Fraud Dataset, 1,000,000 transactions
Algorithm: XGBoost with 10:1 undersampling to handle class imbalance
AUC-PR: 0.55 on balanced validation set
Top features by importance: housing_status, device_os, has_other_cards, keep_alive_session, phone_home_valid

I chose undersampling over scale_pos_weight because it gave significantly better AUC-PR. The dataset has a 90:1 class imbalance so the model needs to see balanced examples during training to learn fraud patterns properly.

## Features

Transaction Analysis: Describe any payment or account application and get a fraud risk score with explanation

Session History: Every conversation in the session is saved and viewable with timestamps and fraud verdicts

Risk Reports: Structured report for the last analysed transaction including model confidence, risk flags, recommended action, and a disclaimer

Screenshot Upload: Upload a payment notification image and the vision LLM extracts the details automatically

Prevention Advice: Ask about any fraud scenario and get specific actionable advice

Fraud Education: Ask general questions about how fraud works, phishing, account takeover, and more

## Tech Stack

Backend: Python, Flask, Groq API (Llama 3.3-70B for chat, Llama 4 Scout for vision)
ML: XGBoost, scikit-learn, pandas
Frontend: HTML, CSS, JavaScript, IBM Plex Mono
Deployment: Render.com, Git LFS for model file

## Why these choices

Groq over OpenAI: Groq is free for development and Llama 3.3-70B is genuinely capable for this use case. The latency is also very low which matters for a real time assistant.

Render over Vercel: Flask needs a persistent server process. Vercel is serverless and would not work for a Flask app that loads a 3MB model at startup.

Undersampling over SMOTE: SMOTE generates synthetic fraud samples which can introduce noise. Undersampling keeps only real data and gave better results on this dataset.

## How to run locally
```bash
git clone https://github.com/Divyansh21k/Fraud_assistant
cd Fraud_assistant
pip install -r requirements.txt
```

Add a `.env` file with your Groq API key:
```
GROQ_API_KEY=your_key_here
```

Get a free key at console.groq.com
```bash
PORT=5001 python3 app.py
```

Then open http://127.0.0.1:5001

## Phase 1

This is Phase 2 of a two phase project. Phase 1 was competing on the IEEE-CIS Fraud Detection Kaggle competition where I built 6 notebooks covering EDA, feature engineering, modeling, tuning, and SHAP interpretability, ending with a public score of 0.8495.

[Phase 1 Notebooks](https://github.com/Divyansh21k/Fraud_detection_ml)
