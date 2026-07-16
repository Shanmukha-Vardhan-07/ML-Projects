"""
Bangalore House Price Predictor — Flask backend
---------------------------------------------------
Loads the trained scikit-learn pipeline (bangalore_house_price_model.pkl)
and the training feature order (columns.pkl), builds the one-hot encoded
feature vector for an incoming request, and returns a predicted price
(in Lakhs INR).
"""

import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load model artifacts
# ---------------------------------------------------------------------------
model = joblib.load("bangalore_house_price_model.pkl")
FEATURE_COLUMNS = joblib.load("columns.pkl")  # exact column order used at train time

# The dataset's one-hot encoding used drop_first=True, so one location and
# one area_type were absorbed into the baseline (all-zero) row instead of
# getting their own column. We add them back in explicitly so the dropdowns
# show every option the model actually saw during training.
BASELINE_LOCATION = "1st Phase JP Nagar"
BASELINE_AREA_TYPE = "Built-up  Area"

LOCATIONS = sorted(
    [BASELINE_LOCATION]
    + [c.replace("location_", "", 1) for c in FEATURE_COLUMNS if c.startswith("location_")]
)

AREA_TYPES = sorted(
    [BASELINE_AREA_TYPE]
    + [c.replace("area_type_", "", 1) for c in FEATURE_COLUMNS if c.startswith("area_type_")]
)


# ---------------------------------------------------------------------------
# Feature engineering helper
# ---------------------------------------------------------------------------
def build_feature_row(total_sqft: float, bath: float, balcony: float, bhk: int,
                       location: str, area_type: str) -> pd.DataFrame:
    """Builds a single-row DataFrame matching the exact column order the
    model was trained on (numeric features + one-hot location/area_type)."""
    row = {col: 0 for col in FEATURE_COLUMNS}

    row["total_sqft"] = total_sqft
    row["bath"] = bath
    row["balcony"] = balcony
    row["bhk"] = bhk

    loc_col = f"location_{location}"
    if loc_col in row:
        row[loc_col] = 1
    # if location == BASELINE_LOCATION, every location_* column stays 0 — correct.

    area_col = f"area_type_{area_type}"
    if area_col in row:
        row[area_col] = 1
    # if area_type == BASELINE_AREA_TYPE, every area_type_* column stays 0 — correct.

    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def predict_price(total_sqft: float, bath: float, balcony: float, bhk: int,
                   location: str, area_type: str) -> float:
    X = build_feature_row(total_sqft, bath, balcony, bhk, location, area_type)
    prediction = model.predict(X)[0]
    return round(max(float(prediction), 0.0), 2)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def home():
    return render_template(
        "index.html",
        locations=LOCATIONS,
        area_types=AREA_TYPES,
        prediction=None,
        error=None,
        form=None,
    )


@app.route("/predict", methods=["POST"])
def predict():
    form_data = {
        "total_sqft": request.form.get("total_sqft", type=float),
        "bath": request.form.get("bath", type=int),
        "balcony": request.form.get("balcony", type=int),
        "bhk": request.form.get("bhk", type=int),
        "location": request.form.get("location", "").strip(),
        "area_type": request.form.get("area_type", ""),
    }

    error = None
    prediction = None
    try:
        if any(v is None or v == "" for v in form_data.values()):
            raise ValueError("all fields are required")
        prediction = predict_price(**form_data)
    except Exception as exc:  # noqa: BLE001
        error = f"Could not generate a prediction: {exc}"

    return render_template(
        "index.html",
        locations=LOCATIONS,
        area_types=AREA_TYPES,
        prediction=prediction,
        error=error,
        form=form_data,
    )


# JSON API endpoint (useful for programmatic access / testing with curl or JS)
@app.route("/api/predict", methods=["POST"])
def predict_api():
    total_sqft = request.form.get("total_sqft", type=float)
    bath = request.form.get("bath", type=int)
    balcony = request.form.get("balcony", type=int)
    bhk = request.form.get("bhk", type=int)
    location = request.form.get("location", "").strip()
    area_type = request.form.get("area_type", "")

    price = predict_price(total_sqft, bath, balcony, bhk, location, area_type)
    return jsonify({"predicted_price_lakhs": price})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)