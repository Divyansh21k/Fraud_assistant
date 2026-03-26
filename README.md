# FraudGuard — UPI Fraud Simulation & Explanation Tool

FraudGuard is now positioned as a **decision-support tool for fintech fraud analysts and support operations teams**.
It turns core transaction signals (amount, time, device/location change) into a clear fraud decision package: **risk score, risk level, key signals, explanation, and recommended actions**.

## Product Positioning (2–3 lines)
- Built for fraud analysts reviewing UPI-style transfers under time pressure.
- Converts model output into actionable decisions, not just technical probabilities.
- Helps teams decide: allow, verify, hold, or escalate.

## Core Workflow
1. Input transaction details.
2. Run risk analysis (`/api/analyze`).
3. Review score + explanation + action layer.
4. Decide and execute controls.

## Structured API Output
`POST /api/analyze`

```json
{
  "risk_score": 82,
  "risk_level": "High",
  "signals": ["Location shift detected from normal usage pattern."],
  "explanation": "This transaction shows ...",
  "actions": [
    "Block transaction temporarily and place hold for 30 minutes.",
    "Trigger step-up verification (OTP + device binding check).",
    "Escalate immediately to manual fraud analyst review."
  ]
}
```

## LLM Prompt Template
Stored in `app.py` as `LLM_PROMPT_TEMPLATE`.

```text
You are FraudGuard Explain, an expert fraud analyst assistant for UPI-style payment reviews.

Return exactly 1 concise paragraph (60-90 words) that is specific and decision-oriented.
Use this case data:
- Amount: {amount}
- Time bucket: {time_bucket}
- Device changed: {device_change}
- Location changed: {location_change}
- Risk score: {risk_score}/100
- Risk level: {risk_level}
- Key signals: {signals}

Rules:
1) Explain why this specific transaction is risky or not.
2) Mention at least two concrete signals.
3) End with one action-focused sentence aligned to risk level.
4) Do not use generic advice or disclaimers.
```

## Wow Feature Included
**Fraud scenario templates** in UI:
- Normal transfer
- UPI collect scam
- Account takeover
- Card testing burst

## Run Locally
```bash
pip install -r requirements.txt
export GROQ_API_KEY=your_key_here
python app.py
```

Open: `http://127.0.0.1:5000`
