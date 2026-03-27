from flask import Flask, request, jsonify
from groq import Groq
from dotenv import load_dotenv
from datetime import datetime
import os

from model import score_transaction

load_dotenv(override=True)

app = Flask(__name__)

GROQ_API_KEY = os.getenv('GROQ_API_KEY')
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

PRODUCT_POSITIONING = [
    'FraudGuard is a decision-support copilot for fintech fraud analysts reviewing UPI-style transfers.',
    'It converts a few transaction signals into a calibrated risk score, clear rationale, and next-best action.',
    'Goal: reduce false negatives without overwhelming support teams with vague alerts.',
]

LLM_PROMPT_TEMPLATE = """You are FraudGuard Explain, an expert fraud analyst assistant for UPI-style payment reviews.

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
"""

SCENARIO_TEMPLATES = {
    'normal_transfer': {
        'name': 'Normal transfer',
        'amount': 2200,
        'time': '14:15',
        'device_change': False,
        'location_change': False,
    },
    'upi_scam': {
        'name': 'UPI collect scam',
        'amount': 18500,
        'time': '23:40',
        'device_change': True,
        'location_change': True,
    },
    'account_takeover': {
        'name': 'Account takeover',
        'amount': 42000,
        'time': '02:10',
        'device_change': True,
        'location_change': False,
    },
    'card_testing': {
        'name': 'Card testing burst',
        'amount': 299,
        'time': '03:05',
        'device_change': True,
        'location_change': True,
    },
}


def parse_hour(time_value: str) -> int:
    try:
        return datetime.strptime(time_value, '%H:%M').hour
    except (TypeError, ValueError):
        return 12


def parse_minutes(time_value: str) -> int:
    try:
        parsed = datetime.strptime(time_value, '%H:%M')
        return parsed.hour * 60 + parsed.minute
    except (TypeError, ValueError):
        return 12 * 60


def time_bucket_from_hour(hour: int) -> str:
    if 0 <= hour < 6:
        return 'Night (00:00-05:59)'
    if 6 <= hour < 12:
        return 'Morning (06:00-11:59)'
    if 12 <= hour < 18:
        return 'Afternoon (12:00-17:59)'
    return 'Evening (18:00-23:59)'


def safe_float(value, default=0.0):
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def derive_behavior_features(payload, prior_transactions=None):
    payload = payload if isinstance(payload, dict) else {}
    prior_transactions = prior_transactions or []

    amount = safe_float(payload.get('amount', 0), 0)
    current_minute = parse_minutes(payload.get('time', '12:00'))

    prior_amounts = [safe_float((txn or {}).get('amount', 0), 0) for txn in prior_transactions]
    baseline_amount = (sum(prior_amounts) / len(prior_amounts)) if prior_amounts else max(amount, 1.0)
    amount_deviation = abs(amount - baseline_amount) / (baseline_amount + 1)

    recent_txn_count = 0
    for txn in prior_transactions:
        delta_minutes = current_minute - parse_minutes((txn or {}).get('time', '12:00'))
        if 0 <= delta_minutes <= 60:
            recent_txn_count += 1
    transaction_velocity = recent_txn_count + 1

    prior_hours = [parse_hour((txn or {}).get('time', '12:00')) for txn in prior_transactions]
    baseline_hour = (sum(prior_hours) / len(prior_hours)) if prior_hours else 13
    current_hour = parse_hour(payload.get('time', '12:00'))
    hour_distance = abs(current_hour - baseline_hour)
    time_anomaly = min(1.0, hour_distance / 12)

    return {
        'amount_deviation': round(float(amount_deviation), 4),
        'transaction_velocity': int(transaction_velocity),
        'time_anomaly': round(float(time_anomaly), 4),
    }


def to_model_features(payload, derived_features):
    payload = payload if isinstance(payload, dict) else {}
    derived_features = derived_features if isinstance(derived_features, dict) else {}
    amount = safe_float(payload.get('amount', 0), 0)
    txn_time = payload.get('time', '12:00')
    device_change = bool(payload.get('device_change', False))
    location_change = bool(payload.get('location_change', False))

    hour = parse_hour(txn_time)
    is_night = 1 if hour < 6 else 0

    return {
        'intended_balcon_amount': amount,
        'velocity_6h': 800 + (amount * 0.12) + (900 * is_night) + (1200 if device_change else 0),
        'velocity_24h': 1800 + (amount * 0.2),
        'foreign_request': 1 if location_change else 0,
        'keep_alive_session': 0 if device_change else 1,
        'credit_risk_score': 95 if (device_change and location_change) else 145,
        'device_fraud_count': 1 if device_change else 0,
        'source': 1 if device_change else 0,
        'amount_deviation': safe_float(derived_features.get('amount_deviation', 0), 0),
        'transaction_velocity': int(safe_float(derived_features.get('transaction_velocity', 1), 1)),
        'time_anomaly': safe_float(derived_features.get('time_anomaly', 0), 0),
        'txn_hour': hour,
    }


def generate_explanation(case_context):
    prompt = LLM_PROMPT_TEMPLATE.format(**case_context)

    if not groq_client:
        return (
            f"This transfer is rated {case_context['risk_score']}/100 ({case_context['risk_level']}) because "
            f"it combines {case_context['signals']}. Prioritize the recommended control action before settlement."
        )

    try:
        response = groq_client.chat.completions.create(
            model='llama-3.3-70b-versatile',
            messages=[
                {'role': 'system', 'content': 'You write precise fraud-review explanations for operations teams.'},
                {'role': 'user', 'content': prompt},
            ],
            temperature=0.2,
            max_tokens=180,
        )
        text = response.choices[0].message.content.strip()
        return text or 'Explanation unavailable. Use risk signals and actions for immediate decision.'
    except Exception as exc:
        print(f"[generate_explanation] Groq call failed: {exc}")
        return 'Explanation service temporarily unavailable. Use key signals and action list for decisioning.'


def build_transaction_response(payload, prior_transactions=None):
    payload = payload if isinstance(payload, dict) else {}
    print(f"[build_transaction_response] Payload: {payload}")
    amount = safe_float(payload.get('amount', 0), 0)
    txn_time = payload.get('time', '12:00')
    device_change = bool(payload.get('device_change', False))
    location_change = bool(payload.get('location_change', False))

    derived_features = derive_behavior_features(payload, prior_transactions=prior_transactions)
    model_features = to_model_features(payload, derived_features)
    print(f"[build_transaction_response] Model features: {model_features}")
    try:
        model_result = score_transaction(model_features)
    except Exception as exc:
        print(f"[build_transaction_response] score_transaction failed: {exc}")
        model_result = {
            'risk_score': 0.0,
            'risk_level': 'LOW',
            'signals': ['Scoring service unavailable; using fail-safe response.'],
            'actions': ['Allow transaction with passive monitoring.'],
        }

    hour = parse_hour(txn_time)
    context = {
        'amount': f'₹{amount:,.2f}',
        'time_bucket': time_bucket_from_hour(hour),
        'device_change': 'Yes' if device_change else 'No',
        'location_change': 'Yes' if location_change else 'No',
        'risk_score': safe_float(model_result.get('risk_score', 0), 0),
        'risk_level': model_result.get('risk_level', 'LOW'),
        'signals': '; '.join(model_result.get('signals', []) or []),
    }

    explanation = generate_explanation(context)

    risk_score = round(max(0.0, min(100.0, safe_float(model_result.get('risk_score', 0), 0))), 2)
    risk_level = model_result.get('risk_level', 'LOW')
    if risk_level not in {'LOW', 'MEDIUM', 'HIGH'}:
        risk_level = 'LOW'

    return {
        'risk_score': risk_score,
        'risk_level': risk_level,
        'signals': model_result.get('signals', []) or [],
        'explanation': explanation or '',
        'actions': model_result.get('actions', []) or [],
    }


@app.route('/')
def index():
    return {'status': 'FraudGuard running'}


@app.route('/health')
def health():
    return {'status': 'ok'}, 200


@app.route('/api/positioning')
def positioning():
    return jsonify({'positioning': PRODUCT_POSITIONING})


@app.route('/api/scenarios')
def scenarios():
    return jsonify({'scenarios': SCENARIO_TEMPLATES})


@app.route('/api/analyze', methods=['POST'])
def analyze():
    payload = request.get_json(silent=True) or {}
    print(f"[analyze] Incoming payload: {payload}")
    if not isinstance(payload, dict):
        payload = {}
    return jsonify(build_transaction_response(payload))


@app.route('/analyze/batch', methods=['POST'])
def analyze_batch():
    payload = request.get_json(silent=True)
    transactions = payload if isinstance(payload, list) else (payload or {}).get('transactions', [])
    print(f"[analyze_batch] Incoming payload type={type(payload).__name__}, count={len(transactions) if isinstance(transactions, list) else 'invalid'}")

    if not isinstance(transactions, list):
        return jsonify({'error': 'Payload must be a list or {"transactions": [...]}'}), 400
    if not transactions:
        return jsonify({'results': [], 'summary': {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}}), 200

    results = []
    summary = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}

    for index, txn in enumerate(transactions):
        if not isinstance(txn, dict):
            print(f"[analyze_batch] Skipping invalid transaction at index {index}: {txn}")
            fallback_result = {
                'risk_score': 0.0,
                'risk_level': 'LOW',
                'signals': ['Invalid transaction payload at this index.'],
                'explanation': 'Transaction skipped due to invalid payload format.',
                'actions': ['Fix payload format and retry.'],
            }
            results.append(fallback_result)
            summary['LOW'] += 1
            continue

        txn = txn or {}
        prior_transactions = transactions[:index]
        result = build_transaction_response(txn, prior_transactions=prior_transactions)
        results.append(result)
        level = result.get('risk_level', 'LOW')
        if level not in summary:
            level = 'LOW'
        summary[level] += 1

    return jsonify({'results': results, 'summary': summary})


@app.errorhandler(Exception)
def global_exception_handler(exc):
    print("ERROR:", str(exc))
    return jsonify({'error': 'Internal Server Error', 'detail': str(exc)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
