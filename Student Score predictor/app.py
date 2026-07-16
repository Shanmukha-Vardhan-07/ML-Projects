from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load model and scaler
model = joblib.load("student_score_model.pkl")
scaler = joblib.load("scaler.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get values from the form
        studytime = float(request.form["studytime"])
        G2 = float(request.form["G2"])
        G1 = float(request.form["G1"])
        Medu = float(request.form["Medu"])
        Fedu = float(request.form["Fedu"])

        # Create DataFrame with the SAME column order
        data = pd.DataFrame({
            "studytime": [studytime],
            "G2": [G2],
            "G1": [G1],
            "Medu": [Medu],
            "Fedu": [Fedu]
        })

        # Scale the input
        data_scaled = scaler.transform(data)

        # Predict
        prediction = model.predict(data_scaled)

        return render_template(
            "index.html",
            prediction_text=f"Predicted Final Grade (G3): {prediction[0]:.2f}"
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Error: {e}"
        )


if __name__ == "__main__":
    app.run(debug=True)