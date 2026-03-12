# 🛡️ FraudGuard — AI Fraud Detection Assistant

### [🚀 Try it Live](https://fraud-assistant.onrender.com) | [📓 Notebook Repo](https://github.com/Divyansh21k/Fraud_detection_ml)

## About FraudGuard

FraudGuard is a conversational fraud detection assistant powered by a real machine learning model trained on 1,000,000 bank account transactions. You describe a transaction in plain English and get an instant fraud risk assessment with explanation.

This is Phase 2 of a two phase fraud detection project. In Phase 1 I competed on the IEEE-CIS Kaggle competition and scored 0.8495. After that I realized the IEEE features are proprietary and useless in conversation, so I retrained on the NeurIPS 2022 Bank Account Fraud Dataset which has fully human readable features and built this assistant around it.

## Why FraudGuard

Most fraud detection models live inside Jupyter notebooks. I wanted to build something real that anyone could actually use. The challenge was bridging the gap between ML features and plain English — FraudGuard solves that by using an LLM to extract features from natural language and pass them to the model.

## ✨ Features

### Core Functionality

🤖 **Transaction Analysis**
- Describe any transaction in plain English
- XGBoost model scores it instantly
- LLM explains what the model found and why
- Risk flag with recommended action (Monitor / Review / Block)

📊 **Risk Reports**
- Structured report for every analysed transaction
- Model confidence, risk level, recommended action
- Full list of detected risk flags
- Disclaimer explaining model limitations honestly

🕐 **Session History**
- Every conversation saved with timestamps
- Fraud and legitimate badges on each entry
- Model confidence shown for scored transactions
- Click any entry to jump back to that point

📸 **Screenshot Upload**
- Upload any payment notification image
- Vision LLM extracts transaction details automatically
- Analysed exactly like a text description

🌍 **Multi-language Support**
- Responds in whatever language you write in
- Switch languages mid conversation

💡 **Fraud Education**
- Ask anything about how fraud works
- Phishing, account takeover, card not present fraud
- Specific prevention advice for your situation

## 📁 Project Structure
```
Fraud_assistant/
├── app.py                  # Flask backend, Groq API integration, chat routes
├── model.py                # Loads ML model, scores transactions, returns risk verdict
├── fraudguard_v2.pkl       # Trained XGBoost model stored via Git LFS
├── requirements.txt        # Python dependencies
├── templates/
│   └── index.html          # Bloomberg terminal UI, History and Reports panels
└── static/
    └── style.css           # Additional styles
```

## 🧠 The ML Model

| Property | Detail |
|---|---|
| Dataset | NeurIPS 2022 Bank Account Fraud, 1,000,000 transactions |
| Algorithm | XGBoost |
| Class imbalance handling | 10:1 undersampling |
| AUC-PR | 0.55 on balanced validation |
| Top features | housing_status, device_os, has_other_cards, keep_alive_session, phone_home_valid |
| Decision threshold | 10% — flags for review rather than hard block |

The model flags suspicious transactions for human review rather than making a definitive verdict. This is intentional — real fraud systems work this way because false positives have a real cost.

## 🛠️ Tech Stack

**Backend**
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=flat&logo=flask&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-FF6600?style=flat)
![Groq](https://img.shields.io/badge/Groq-FF6600?style=flat)

**Frontend**
![HTML](https://img.shields.io/badge/HTML5-E34F26?style=flat&logo=html5&logoColor=white)
![CSS](https://img.shields.io/badge/CSS3-1572B6?style=flat&logo=css3&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=flat&logo=javascript&logoColor=black)

**Deployment**
![Render](https://img.shields.io/badge/Render-46E3B7?style=flat&logo=render&logoColor=white)
![Git LFS](https://img.shields.io/badge/Git_LFS-F05032?style=flat&logo=git&logoColor=white)

## 🚀 How to Run Locally
```bash
git clone https://github.com/Divyansh21k/Fraud_assistant
cd Fraud_assistant
pip install -r requirements.txt
```

Create a `.env` file:
```
GROQ_API_KEY=your_key_here
```

Get a free key at console.groq.com
```bash
PORT=5001 python3 app.py
```

Open http://127.0.0.1:5001

## 💡 Key Decisions

**Why Groq over OpenAI** — Groq is free for development and Llama 3.3-70B is genuinely capable for this use case. The latency is also very low which matters for a real time assistant.

**Why Render over Vercel** — Flask needs a persistent server process. Vercel is serverless and does not work for a Flask app that loads a 3MB model at startup.

**Why undersampling over scale_pos_weight** — Undersampling gave significantly better AUC-PR on this dataset. The 90:1 class imbalance meant scale_pos_weight was overcorrecting and hurting precision.

**Why flag rather than verdict** — The model has 0.55 AUC-PR on balanced data. Presenting its output as a hard verdict would be dishonest. Flagging for review is how real fraud systems work and is the right framing here.

## 📓 Phase 1

This is Phase 2 of a two phase project. Phase 1 was competing on the IEEE-CIS Fraud Detection Kaggle competition where I built 6 notebooks covering EDA, feature engineering, modeling, tuning, and SHAP interpretability, ending with a public score of 0.8495.

[Phase 1 Notebooks](https://github.com/Divyansh21k/Fraud_detection_ml)

Made with ❤️ by Divyansh Kharnal
