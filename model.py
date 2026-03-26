import pickle
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
    'month': 3,
}


def get_risk_level(risk_score: float) -> str:
    if risk_score >= 75:
        return 'High'
    if risk_score >= 45:
        return 'Medium'
    return 'Low'


def get_actions(risk_level: str):
    action_map = {
        'High': [
            'Block transaction temporarily and place hold for 30 minutes.',
            'Trigger step-up verification (OTP + device binding check).',
            'Escalate immediately to manual fraud analyst review.',
        ],
        'Medium': [
            'Trigger OTP verification before settlement.',
            'Add transaction to analyst queue for same-day review.',
            'Notify customer in-app and request confirmation of intent.',
        ],
        'Low': [
            'Allow transaction with passive monitoring.',
            'Log case for behavior baseline updates.',
            'No manual review required unless repeated anomalies appear.',
        ],
    }
    return action_map.get(risk_level, action_map['Medium'])


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
    risk_score = round(probability * 100, 2)
    risk_level = get_risk_level(risk_score)

    flags = []
    if transaction_info.get('foreign_request', 0) == 1:
        flags.append('Location shift detected from normal usage pattern.')
    if transaction_info.get('velocity_6h', 0) > 2500:
        flags.append('High transfer velocity in a short window.')
    if transaction_info.get('credit_risk_score', 150) < 100:
        flags.append('Customer profile indicates elevated baseline risk.')
    if transaction_info.get('device_fraud_count', 0) > 0:
        flags.append('Device has prior fraud-linked history.')
    if transaction_info.get('keep_alive_session', 1) == 0:
        flags.append('Session pattern appears abrupt and atypical.')

    if not flags:
        flags.append('No major anomaly signal triggered beyond baseline behavior.')

    return {
        'risk_score': risk_score,
        'risk_level': risk_level,
        'signals': flags,
        'actions': get_actions(risk_level),
        'threshold_used': round(float(threshold) * 100, 2),
    }
