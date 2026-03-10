import pickle
import numpy as np
import pandas as pd

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

def score_transaction(transaction_info):
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

    probability = float(model.predict_proba(df)[0][1])
    print(f"DEBUG probability: {probability:.4f}, threshold: {threshold:.4f}")

    verdict = "FRAUD" if probability >= threshold else "LEGITIMATE"

    if probability >= 0.30:
        risk_level = "Critical — Recommend Block"
    elif probability >= 0.20:
        risk_level = "High — Recommend Review"
    elif probability >= 0.10:
        risk_level = "Medium — Monitor"
    else:
        risk_level = "Low — Clear"
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

    return {
        'probability': round(probability * 100, 2),
        'verdict': verdict,
        'risk_level': risk_level,
        'flags': flags,
        'threshold_used': round(float(threshold) * 100, 2)
    }