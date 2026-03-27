from fastapi import Body, FastAPI, Request
from fastapi.responses import JSONResponse
from typing import Any, Dict, List

app = FastAPI(title="FraudGuard — UPI Fraud Simulation & Decision Engine")


# -----------------------------
# Safe parsing helper functions
# -----------------------------
def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def to_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n"}:
            return False
    if isinstance(value, (int, float)):
        return value != 0
    return default


def to_hour(value: Any, default: int = 12) -> int:
    hour = int(to_float(value, default))
    if hour < 0:
        return 0
    if hour > 23:
        return 23
    return hour


# ---------------------------------
# Fraud scoring and decision engine
# ---------------------------------
def simulated_model_score(amount: float, avg_amount: float, hour: int, device_changed: bool, location_changed: bool) -> float:
    """Simulated ML-like score (no external model): simple weighted heuristic."""
    score = 0.0

    # Model intuition: anomaly vs historical average
    if avg_amount > 0:
        ratio = amount / avg_amount
        if ratio >= 3.0:
            score += 22
        elif ratio >= 2.0:
            score += 15
        elif ratio >= 1.5:
            score += 8
    elif amount > 0:
        # Missing historical average but non-zero amount still gets a small uncertainty boost
        score += 6

    if amount > 100000:
        score += 20
    elif amount > 50000:
        score += 12
    elif amount > 20000:
        score += 7

    if device_changed:
        score += 12
    if location_changed:
        score += 12

    if hour >= 0 and hour <= 5:
        score += 10

    return min(100.0, max(0.0, score))


def rules_score_and_signals(amount: float, avg_amount: float, hour: int, device_changed: bool, location_changed: bool) -> Dict[str, Any]:
    """Rule engine required by product spec."""
    score = 0.0
    signals: List[str] = []

    # amount spike vs avg → +25
    if avg_amount > 0 and amount >= (avg_amount * 2):
        score += 25
        signals.append("Amount spike detected versus average transaction value")

    # high value (>50000) → +20
    if amount > 50000:
        score += 20
        signals.append("High-value transaction above 50,000")

    # device change → +25
    if device_changed:
        score += 25
        signals.append("Transaction initiated from a changed device")

    # location change → +25
    if location_changed:
        score += 25
        signals.append("Transaction initiated from a changed location")

    # night transaction → +15
    if 0 <= hour <= 5:
        score += 15
        signals.append("Night-time transaction pattern observed")

    return {"score": score, "signals": signals}


def classify_risk(score: float) -> str:
    if score <= 40:
        return "LOW"
    if score < 70:
        return "MEDIUM"
    return "HIGH"


def recommended_actions(risk_level: str) -> List[str]:
    if risk_level == "HIGH":
        return ["block transaction", "OTP verification", "manual review"]
    if risk_level == "MEDIUM":
        return ["OTP verification", "monitor"]
    return ["allow"]


def generate_explanation(signals: List[str], risk_level: str, amount: float, avg_amount: float) -> str:
    # LLM-style narrative without external API
    if not signals:
        return (
            "This transaction aligns with the usual user behavior and does not show strong fraud indicators, "
            "so fraud likelihood appears low and the transaction can proceed with standard controls."
        )

    top_reasons = ", ".join(signals[:3])
    if risk_level == "HIGH":
        return (
            "This transaction is flagged due to "
            f"{top_reasons}, creating a high-risk combination that strongly indicates potential fraud. "
            "Immediate blocking, OTP challenge, and analyst review are recommended before completion."
        )

    if risk_level == "MEDIUM":
        return (
            "This transaction shows moderate fraud indicators, including "
            f"{top_reasons}, which raises concern compared with expected behavior. "
            "OTP verification and close monitoring are recommended to reduce fraud exposure."
        )

    return (
        "This transaction has limited risk indicators"
        f" ({top_reasons}) and remains within a low-risk profile overall. "
        "It can be allowed while maintaining normal monitoring safeguards."
    )


def sanitize_transaction_input(raw_payload: Any) -> Dict[str, Any]:
    payload = raw_payload if isinstance(raw_payload, dict) else {}

    return {
        "amount": to_float(payload.get("amount", 0.0), 0.0),
        "device_changed": to_bool(payload.get("device_changed", False), False),
        "location_changed": to_bool(payload.get("location_changed", False), False),
        "hour": to_hour(payload.get("hour", 12), 12),
        "avg_amount": to_float(payload.get("avg_amount", 0.0), 0.0),
    }


def analyze_one(raw_payload: Any) -> Dict[str, Any]:
    txn = sanitize_transaction_input(raw_payload)

    amount = txn.get("amount", 0.0)
    avg_amount = txn.get("avg_amount", 0.0)
    hour = txn.get("hour", 12)
    device_changed = txn.get("device_changed", False)
    location_changed = txn.get("location_changed", False)

    model_score = simulated_model_score(amount, avg_amount, hour, device_changed, location_changed)
    rule_output = rules_score_and_signals(amount, avg_amount, hour, device_changed, location_changed)
    rule_score = to_float(rule_output.get("score", 0.0), 0.0)
    signals = rule_output.get("signals", []) if isinstance(rule_output.get("signals", []), list) else []

    final_score = min(100.0, max(0.0, model_score + rule_score))
    risk_score = round(final_score, 2)
    risk_level = classify_risk(risk_score)

    # Ensure non-empty, no None values in output
    if not signals:
        signals = ["No major fraud signals triggered by current rules"]

    actions = recommended_actions(risk_level)
    explanation = generate_explanation(signals, risk_level, amount, avg_amount)

    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "signals": [str(s) for s in signals if s is not None],
        "explanation": str(explanation or "Risk explanation unavailable"),
        "actions": [str(a) for a in actions if a is not None],
    }


# ----------
# Endpoints
# ----------
@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "status": "FraudGuard running",
        "endpoints": ["/analyze", "/analyze/batch"],
    }


@app.post("/analyze")
def analyze(payload: Any = Body(default=None)) -> Dict[str, Any]:
    try:
        return analyze_one(payload)
    except Exception:
        # Fail-safe response to guarantee valid JSON and no crash
        return {
            "risk_score": 0.0,
            "risk_level": "LOW",
            "signals": ["Fallback response triggered due to internal processing issue"],
            "explanation": "Transaction processed in safe mode because of an internal validation issue.",
            "actions": ["allow"],
        }


@app.post("/analyze/batch")
def analyze_batch(payload: Any = Body(default=None)) -> Dict[str, Any]:
    try:
        transactions = payload if isinstance(payload, list) else []

        results: List[Dict[str, Any]] = []
        summary = {"high": 0, "medium": 0, "low": 0}

        for item in transactions:
            result = analyze_one(item)
            results.append(result)

            level = result.get("risk_level", "LOW")
            if level == "HIGH":
                summary["high"] += 1
            elif level == "MEDIUM":
                summary["medium"] += 1
            else:
                summary["low"] += 1

        return {
            "results": results,
            "summary": summary,
        }
    except Exception:
        return {
            "results": [],
            "summary": {"high": 0, "medium": 0, "low": 0},
        }


@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    # Global final safety net for valid JSON
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": str(exc) if str(exc) else "Unexpected server error",
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
