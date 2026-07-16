import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "knn_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "model", "label_encoder.pkl")

model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)

FEATURE_NAMES = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]

# Friendly info shown alongside the prediction
SPECIES_INFO = {
    "Iris-setosa": {
        "common_name": "Setosa",
        "fact": "Smallest of the three — short, rounded petals and a compact flower.",
    },
    "Iris-versicolor": {
        "common_name": "Versicolor",
        "fact": "The in-betweener — medium-sized petals and sepals.",
    },
    "Iris-virginica": {
        "common_name": "Virginica",
        "fact": "The largest — long petals and sepals set it apart from the rest.",
    },
}


@app.route("/", methods=["GET"])
def home():
    # Defaults roughly at the dataset's mean values, just so the flower isn't blank
    defaults = {
        "sepal_length": 5.8,
        "sepal_width": 3.0,
        "petal_length": 3.8,
        "petal_width": 1.2,
    }
    return render_template("index.html", result=None, values=defaults)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        sepal_length = float(request.form["sepal_length"])
        sepal_width = float(request.form["sepal_width"])
        petal_length = float(request.form["petal_length"])
        petal_width = float(request.form["petal_width"])
    except (KeyError, ValueError):
        return render_template(
            "index.html",
            result={"error": "Please enter valid numbers for all four measurements."},
            values={
                "sepal_length": 5.8,
                "sepal_width": 3.0,
                "petal_length": 3.8,
                "petal_width": 1.2,
            },
        )

    values = {
        "sepal_length": sepal_length,
        "sepal_width": sepal_width,
        "petal_length": petal_length,
        "petal_width": petal_width,
    }

    # Build input as a DataFrame with the same column names used during training
    input_df = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                            columns=FEATURE_NAMES)

    prediction = model.predict(input_df)[0]

    # The model was trained directly on species names, so prediction is already
    # a label like "Iris-setosa". label_encoder is kept here for consistency /
    # in case you swap in a model trained on encoded labels later.
    species = prediction if isinstance(prediction, str) else label_encoder.inverse_transform([prediction])[0]

    # Confidence, if the underlying model supports predict_proba (KNN does)
    confidence = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_df)[0]
        classes = list(model.classes_)
        confidence = round(float(proba[classes.index(species)]) * 100, 1)

    info = SPECIES_INFO.get(species, {"common_name": species, "fact": ""})

    result = {
        "species": species,
        "common_name": info["common_name"],
        "fact": info["fact"],
        "confidence": confidence,
    }

    return render_template("index.html", result=result, values=values)


if __name__ == "__main__":
    app.run(debug=True)