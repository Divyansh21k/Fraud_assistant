import pickle
import numpy as np
import pandas as pd
from datetime import datetime

with open('fraudguard_v2.pkl', 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
threshold = model_data['threshold']
feature_columns = model_data['feature_columns']

FEATURE_DEFAULTS = {
    'income': 0.5,
    'name_email_similarity': 0.5,
    'prev_address_months_count': 12,
    'current_address_months_count': 24,
    'customer_age': 35,
    'days_since_request': 1,
    'intended_balcon_amount': 0,
    'payment_type': 2,
    'velocity_6h': 500,
    'velocity_24h': 1000,
    'velocity_4w': 2000,
    'zip_count_4w': 200,
    'date_of_birth_distinct_emails_4w': 3,
    'bank_branch_count_8w': 10,
    'employment_status': 1,
    'credit_risk_score': 150,
    'email_is_free': 0,
    'housing_status': 2,
    'phone_home_valid': 1,
    'phone_mobile_valid': 1,
    'bank_months_count': 24,
    'has_other_cards': 0,
    'proposed_credit_limit': 1000,
    'foreign_request': 0,
    'source': 0,
    'session_length_in_minutes': 5,
    'device_os': 2,
    'keep_alive_session': 1,
    'device_distinct_emails_8w': 1,
    'device_fraud_count': 0,
    'month': 3
}


def _rule_score(info):
    """
    Rule-based scoring layer. Returns a score 0-100 and list of triggered signals.
    These rules are inspired by real fintech risk systems.
    """
    score = 0
    signals = []

    amount = info.get('amount', 0) or 0
    avg_amount = info.get('avg_amount', amount) or amount
    hour = info.get('hour', 12) or 12
    device_changed = info.get('device_changed', False)
    location_changed = info.get('location_changed', False)

    # High value transaction
    if amount > 50000:
        score += 20
        signals.append(f"high-value transaction (₹{amount:,.0f})")

    # Amount spike vs average
    if avg_amount > 0 and amount > 0:
        ratio = amount / avg_amount
        if ratio >= 3:
            score += 25
            signals.append(f"amount {ratio:.1f}x above average")
        elif ratio >= 2:
            score += 12

    # Device change
    if device_changed:
        score += 25
        signals.append("new/unrecognised device")

    # Location change
    if location_changed:
        score += 25
        signals.append("unusual location")

    # Unusual time (11pm - 5am)
    if hour >= 23 or hour <= 5:
        score += 15
        signals.append(f"unusual hour ({hour:02d}:xx)")

    return min(score, 100), signals


def _build_explanation(ml_prob, rule_score_val, flags, signals, verdict):
    """
    Generates a human-readable explanation referencing actual signals.
    No external API — pure logic-driven text.
    """
    parts = []

    if verdict == "FRAUD":
        if ml_prob >= 0.30:
            parts.append("The model has high confidence this transaction is fraudulent.")
        elif ml_prob >= 0.20:
            parts.append("This transaction shows strong fraud indicators.")
        else:
            parts.append("This transaction has been flagged based on combined risk signals.")
    else:
        parts.append("This transaction shows no strong fraud indicators.")

    if signals:
        parts.append("Key risk signals: " + ", ".join(signals) + ".")

    if flags:
        parts.append("Additional flags: " + ", ".join(flags) + ".")

    if verdict == "FRAUD":
        if rule_score_val >= 50:
            parts.append("Both the ML model and risk rules are aligned on this being high risk.")
        else:
            parts.append("The ML model is the primary driver of this flag.")
    else:
        parts.append("No single rule or model signal crossed the review threshold independently.")

    return " ".join(parts)


def _get_actions(risk_level):
    level = risk_level.split("—")[0].strip().lower()
    if level == "critical":
        return ["block transaction immediately", "require OTP re-verification", "escalate to manual review", "notify account holder"]
    elif level == "high":
        return ["require OTP verification", "flag for manual review", "send alert to account holder"]
    elif level == "medium":
        return ["monitor for further suspicious activity", "log for batch review"]
    else:
        return ["allow transaction"]


def score_transaction(transaction_info):
    """
    Hybrid scoring: ML model probability + rule-based score.
    Always returns a valid, complete structured result.
    """
    if not isinstance(transaction_info, dict):
        transaction_info = {}

    # --- ML MODEL SCORE ---
    try:
        features = FEATURE_DEFAULTS.copy()
        features.update({k: v for k, v in transaction_info.items() if k in features})
        df = pd.DataFrame([features])

        if 'velocity_ratio' in feature_columns:
            df['velocity_ratio'] = df['velocity_6h'] / (df['velocity_24h'] + 1)
            df['email_device_risk'] = df['email_is_free'] * df['device_distinct_emails_8w']
            df['age_income_ratio'] = df['customer_age'] / (df['income'] + 0.01)
            df['credit_to_limit_ratio'] = df['credit_risk_score'] / (df['proposed_credit_limit'] + 1)
            df['address_stability'] = df['current_address_months_count'] + df['prev_address_months_count']
            df['phone_trust'] = df['phone_home_valid'] + df['phone_mobile_valid']

        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0
        df = df[feature_columns]

        ml_prob = float(model.predict_proba(df)[0][1])
    except Exception as e:
        print(f"ML scoring error (using fallback): {e}")
        ml_prob = 0.05

    # --- RULE SCORE ---
    rule_val, signals = _rule_score(transaction_info)

    # --- HYBRID SCORE ---
    # ML score normalised to 0-100, then blended 60/40 with rules
    ml_score = ml_prob * 100
    hybrid_score = round(min((ml_score * 0.6) + (rule_val * 0.4), 100), 1)

    # --- VERDICT ---
    verdict = "FRAUD" if ml_prob >= float(threshold) else "LEGITIMATE"

    # Override to FRAUD if rules are very strong and ML is borderline
    if rule_val >= 70 and ml_prob >= (float(threshold) * 0.7):
        verdict = "FRAUD"

    # --- RISK LEVEL ---
    if hybrid_score >= 60 or ml_prob >= 0.30:
        risk_level = "Critical — Recommend Block"
    elif hybrid_score >= 35 or ml_prob >= 0.20:
        risk_level = "High — Recommend Review"
    elif hybrid_score >= 15 or ml_prob >= 0.10:
        risk_level = "Medium — Monitor"
    else:
        risk_level = "Low — Clear"

    # --- FLAGS (model feature flags) ---
    flags = []
    if transaction_info.get('foreign_request', 0) == 1:
        flags.append("foreign request")
    if transaction_info.get('email_is_free', 0) == 1:
        flags.append("free email provider")
    if transaction_info.get('velocity_6h', 1) > 5:
        flags.append("high transaction velocity")
    if transaction_info.get('credit_risk_score', 150) < 100:
        flags.append("low credit risk score")
    if transaction_info.get('device_fraud_count', 0) > 0:
        flags.append("device previously linked to fraud")
    if transaction_info.get('phone_home_valid', 1) == 0 and transaction_info.get('phone_mobile_valid', 1) == 0:
        flags.append("no valid phone number")
    if transaction_info.get('session_length_in_minutes', 5) < 1:
        flags.append("very short session")

    explanation = _build_explanation(ml_prob, rule_val, flags, signals, verdict)
    actions = _get_actions(risk_level)

    print(f"DEBUG ml_prob={ml_prob:.4f} rule={rule_val} hybrid={hybrid_score} verdict={verdict}")

    return {
        'probability': round(ml_prob * 100, 2),
        'hybrid_score': hybrid_score,
        'verdict': verdict,
        'risk_level': risk_level,
        'risk_score': hybrid_score,
        'signals': signals,
        'flags': flags,
        'explanation': explanation,
        'actions': actions,
        'threshold_used': round(float(threshold) * 100, 2)
    }