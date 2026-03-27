from fastapi import Body, FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
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
@app.get("/", response_class=HTMLResponse)
def root() -> HTMLResponse:
    html = """
    <!doctype html>
    <html lang="en">
      <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>FraudGuard Risk Analyzer</title>
        <style>
          :root {
            color-scheme: light dark;
            font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
          }
          body {
            margin: 0;
            padding: 24px;
            background: #f5f7fb;
            color: #1f2937;
          }
          .card {
            max-width: 720px;
            margin: 0 auto;
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.06);
          }
          h1 { margin-top: 0; margin-bottom: 4px; font-size: 1.5rem; }
          .subtitle { margin-top: 0; color: #4b5563; }
          .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 12px;
            margin-bottom: 14px;
          }
          label { font-weight: 600; display: block; margin-bottom: 6px; }
          input[type="number"] {
            width: 100%;
            box-sizing: border-box;
            padding: 10px;
            border: 1px solid #d1d5db;
            border-radius: 8px;
            background: #fff;
            color: #111827;
          }
          .check {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-top: 6px;
          }
          .check label {
            margin: 0;
            font-weight: 500;
          }
          button {
            border: none;
            border-radius: 8px;
            padding: 10px 14px;
            background: #2563eb;
            color: #fff;
            font-weight: 700;
            cursor: pointer;
          }
          button:hover { background: #1d4ed8; }
          .result {
            margin-top: 18px;
            background: #f9fafb;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 14px;
          }
          .result h2 { margin-top: 0; font-size: 1.1rem; }
          .muted { color: #6b7280; }
          ul { margin-top: 6px; margin-bottom: 0; padding-left: 18px; }
          .error { color: #b91c1c; font-weight: 600; margin-top: 10px; }
        </style>
      </head>
      <body>
        <main class="card">
          <h1>FraudGuard</h1>
          <p class="subtitle">Analyze transaction risk instantly.</p>
          <div class="grid">
            <div>
              <label for="amount">Amount</label>
              <input id="amount" type="number" min="0" step="0.01" placeholder="e.g. 25000" />
            </div>
            <div>
              <label for="hour">Hour (0-23)</label>
              <input id="hour" type="number" min="0" max="23" step="1" placeholder="e.g. 2" />
            </div>
          </div>
          <div class="grid">
            <div class="check">
              <input id="device_changed" type="checkbox" />
              <label for="device_changed">Device changed</label>
            </div>
            <div class="check">
              <input id="location_changed" type="checkbox" />
              <label for="location_changed">Location changed</label>
            </div>
          </div>
          <button id="analyzeBtn" type="button">Analyze Risk</button>
          <div id="error" class="error" role="alert" aria-live="polite"></div>
          <section id="result" class="result" aria-live="polite">
            <h2>Result</h2>
            <p class="muted">Submit a transaction to see risk details.</p>
          </section>
        </main>
        <script>
          const amountEl = document.getElementById("amount");
          const hourEl = document.getElementById("hour");
          const deviceChangedEl = document.getElementById("device_changed");
          const locationChangedEl = document.getElementById("location_changed");
          const analyzeBtn = document.getElementById("analyzeBtn");
          const resultEl = document.getElementById("result");
          const errorEl = document.getElementById("error");

          function toNumber(value, fallback = 0) {
            if (value === "" || value === null || value === undefined) return fallback;
            const parsed = Number(value);
            return Number.isFinite(parsed) ? parsed : fallback;
          }

          function toList(items) {
            if (!Array.isArray(items) || items.length === 0) {
              return "<li>None</li>";
            }
            return items.map((item) => `<li>${String(item)}</li>`).join("");
          }

          async function analyzeRisk() {
            errorEl.textContent = "";
            resultEl.innerHTML = "<h2>Result</h2><p class='muted'>Analyzing...</p>";
            const payload = {
              amount: Math.max(0, toNumber(amountEl.value, 0)),
              hour: Math.min(23, Math.max(0, Math.trunc(toNumber(hourEl.value, 12)))),
              device_changed: Boolean(deviceChangedEl.checked),
              location_changed: Boolean(locationChangedEl.checked),
            };

            try {
              const response = await fetch("/analyze", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
              });

              if (!response.ok) {
                throw new Error(`Request failed with status ${response.status}`);
              }

              const data = await response.json();
              resultEl.innerHTML = `
                <h2>Result</h2>
                <p><strong>Risk score:</strong> ${data.risk_score ?? "N/A"}</p>
                <p><strong>Risk level:</strong> ${data.risk_level ?? "N/A"}</p>
                <p><strong>Signals:</strong></p>
                <ul>${toList(data.signals)}</ul>
                <p><strong>Explanation:</strong> ${data.explanation ?? "N/A"}</p>
                <p><strong>Actions:</strong></p>
                <ul>${toList(data.actions)}</ul>
              `;
            } catch (error) {
              resultEl.innerHTML = "<h2>Result</h2><p class='muted'>No result available.</p>";
              errorEl.textContent = "Could not analyze risk right now. Please check your input and try again.";
              console.error(error);
            }
          }

          analyzeBtn.addEventListener("click", analyzeRisk);
        </script>
      </body>
    </html>
    """
    return HTMLResponse(content=html)


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
