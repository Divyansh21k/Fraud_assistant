import os
import pickle
import pandas as pd


def _safe_float(value, default=0.0):
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value, default=0):
    try:
        if value is None:
            return int(default)
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


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

model = None
threshold = 0.5
feature_columns = list(FEATURE_DEFAULTS.keys())

try:
    model_path = os.path.join(os.path.dirname(__file__), 'fraudguard_v2.pkl')
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)

    model = model_data.get('model')
    threshold = _safe_float(model_data.get('threshold', 0.5), 0.5)
    feature_columns = model_data.get('feature_columns') or list(FEATURE_DEFAULTS.keys())
    print(f"[model] Loaded model from {model_path}")
except Exception as exc:
    print(f"[model] Failed to load model: {exc}")


def get_risk_level(risk_score: float) -> str:
    if risk_score >= 75:
        return 'HIGH'
    if risk_score >= 45:
        return 'MEDIUM'
    return 'LOW'


def get_actions(risk_level: str):
    action_map = {
        'HIGH': [
            'Block transaction temporarily and place hold for 30 minutes.',
            'Trigger step-up verification (OTP + device binding check).',
            'Escalate immediately to manual fraud analyst review.',
        ],
        'MEDIUM': [
            'Trigger OTP verification before settlement.',
            'Add transaction to analyst queue for same-day review.',
            'Notify customer in-app and request confirmation of intent.',
        ],
        'LOW': [
            'Allow transaction with passive monitoring.',
            'Log case for behavior baseline updates.',
            'No manual review required unless repeated anomalies appear.',
        ],
    }
    return action_map.get(risk_level, action_map['MEDIUM'])


def compute_rule_signals(transaction_info):
    rule_score = 0
    signals = []
    transaction_info = transaction_info if isinstance(transaction_info, dict) else {}

    amount = _safe_float(transaction_info.get('intended_balcon_amount', 0), 0)
    transaction_velocity = _safe_float(transaction_info.get('transaction_velocity', 1), 1)
    amount_deviation = _safe_float(transaction_info.get('amount_deviation', 0), 0)
    time_anomaly = _safe_float(transaction_info.get('time_anomaly', 0), 0)
    txn_hour = _safe_int(transaction_info.get('txn_hour', 12), 12)

    if amount >= 15000:
        rule_score += 22
        signals.append('High amount compared with regular UPI transfer range.')

    if transaction_info.get('keep_alive_session', 1) == 0:
        rule_score += 18
        signals.append('Device changed recently for this transfer session.')

    if transaction_info.get('foreign_request', 0) == 1:
        rule_score += 18
        signals.append('Location change detected from expected customer region.')

    if txn_hour < 6 or txn_hour >= 23 or time_anomaly >= 0.55:
        rule_score += 14
        signals.append('Transaction time is unusual for this customer behavior pattern.')

    if amount_deviation >= 0.7:
        rule_score += 12
        signals.append('Amount deviates materially from baseline spending behavior.')

    if transaction_velocity >= 4:
        rule_score += 12
        signals.append('High transaction velocity detected in short time window.')

    return rule_score, signals


def score_transaction(transaction_info):
    transaction_info = transaction_info if isinstance(transaction_info, dict) else {}
    print(f"[score_transaction] Input received: {transaction_info}")

    features = FEATURE_DEFAULTS.copy()
    features.update({k: v for k, v in transaction_info.items() if k in features and v is not None})

    df = pd.DataFrame([features])

    if 'velocity_ratio' in feature_columns:
        df['velocity_ratio'] = df['velocity_6h'] / (df['velocity_24h'] + 1)
        df['email_device_risk'] = df['email_is_free'] * df['device_distinct_emails_8w']
        df['age_income_ratio'] = df['customer_age'] / (df['income'] + 0.01)
        df['credit_to_limit_ratio'] = df['credit_risk_score'] / (df['proposed_credit_limit'] + 1)
        df['address_stability'] = df['current_address_months_count'] + df['prev_address_months_count']
        df['phone_trust'] = df['phone_home_valid'] + df['phone_mobile_valid']

    columns = feature_columns or list(FEATURE_DEFAULTS.keys())
    for col in columns:
        if col not in df.columns:
            df[col] = 0
    df = df[columns]

    model_score = 0.0
    if model is not None:
        try:
            probability = _safe_float(model.predict_proba(df)[0][1], 0.0)
            model_score = round(max(0.0, min(1.0, probability)) * 100, 2)
        except Exception as exc:
            print(f"[score_transaction] Model scoring failed: {exc}")
    else:
        print("[score_transaction] Model unavailable; falling back to rules only.")

    rule_score, rule_signals = compute_rule_signals(transaction_info)
    final_score = round(max(0.0, min(100.0, _safe_float(model_score, 0) + _safe_float(rule_score, 0))), 2)
    risk_level = get_risk_level(final_score)

    if not rule_signals:
        rule_signals.append('No major rule-based anomaly triggered beyond baseline behavior.')

    return {
        'risk_score': final_score,
        'risk_level': risk_level,
        'signals': rule_signals,
        'actions': get_actions(risk_level),
        'model_score': _safe_float(model_score, 0),
        'rule_score': _safe_float(rule_score, 0),
        'threshold_used': round(_safe_float(threshold, 0.5) * 100, 2),
    }
