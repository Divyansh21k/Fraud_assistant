import pickle
import numpy as np
import pandas as pd

# load the trained model once when the app starts
with open('final_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
threshold = model_data['threshold']
feature_columns = model_data['feature_columns']

# median values from training data - used when user doesn't provide a value
FEATURE_MEDIANS = {col: 0 for col in feature_columns}

def build_transaction_features(transaction_info):
    """
    Takes a dictionary of transaction details from the user
    and builds a full feature vector the model can score.
    Missing values are filled with median/default values.
    """
    features = FEATURE_MEDIANS.copy()

    # map user provided values to model features
    amount = transaction_info.get('amount', 100)
    hour = transaction_info.get('hour', 12)
    is_mobile = transaction_info.get('is_mobile', 0)
    is_free_email = transaction_info.get('is_free_email', 0)
    email_match = transaction_info.get('email_match', 0)
    has_identity = transaction_info.get('has_identity', 1)

    # engineered features
    features['TransactionAmt'] = amount
    features['log_amount'] = np.log1p(amount)
    features['hour'] = hour
    features['is_late_night'] = 1 if 1 <= hour <= 9 else 0
    features['is_card_probe'] = 1 if amount < 1 else 0
    features['is_threshold_avoid'] = 1 if 500 <= amount < 1000 else 0
    features['is_mobile'] = is_mobile
    features['has_identity'] = has_identity
    features['is_free_email'] = is_free_email
    features['email_match'] = email_match

    # build dataframe with correct column order
    df = pd.DataFrame([features])[feature_columns]
    return df

def score_transaction(transaction_info):
    """
    Score a transaction and return probability and risk level.
    """
    features = build_transaction_features(transaction_info)
    probability = model.predict_proba(features)[0][1]

    if probability >= threshold:
        verdict = "FRAUD"
    else:
        verdict = "LEGITIMATE"

    if probability >= 0.85:
        risk_level = "Critical"
    elif probability >= 0.70:
        risk_level = "High"
    elif probability >= 0.40:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    # figure out which features are suspicious
    flags = []
    if transaction_info.get('hour', 12) in range(1, 10):
        flags.append("late night transaction")
    if transaction_info.get('amount', 100) < 1:
        flags.append("card probe amount")
    if 500 <= transaction_info.get('amount', 100) < 1000:
        flags.append("threshold avoidance amount")
    if transaction_info.get('is_mobile', 0):
        flags.append("mobile device")
    if transaction_info.get('is_free_email', 0):
        flags.append("free email provider")
    if transaction_info.get('email_match', 0):
        flags.append("purchaser and recipient email match")

    return {
        'probability': round(float(probability) * 100, 2),
        'verdict': verdict,
        'risk_level': risk_level,
        'flags': flags,
        'threshold_used': round(threshold * 100, 2)
    }