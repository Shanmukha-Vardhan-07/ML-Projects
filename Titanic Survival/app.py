from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load("titanic_random_forest.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:
        # Get values from form
        pclass = int(request.form["Pclass"])
        sex = request.form["Sex"]
        age = float(request.form["Age"])
        sibsp = int(request.form["SibSp"])
        parch = int(request.form["Parch"])
        fare = float(request.form["Fare"])
        embarked = request.form["Embarked"]

        # Manual Encoding

        # Sex
        sex = 1 if sex == "male" else 0

        # Embarked
        embarked_dict = {
            "C": 0,
            "Q": 1,
            "S": 2
        }

        if embarked not in embarked_dict:
            raise ValueError("Invalid Embarked value")
        embarked = embarked_dict[embarked]

    except (KeyError, ValueError):
        return render_template(
            "index.html",
            prediction_text="⚠️ Please fill in all fields with valid values."
        )

    # Arrange data in the same order as training
    features = np.array([[

        pclass,
        sex,
        age,
        sibsp,
        parch,
        fare,
        embarked

    ]])

    # Prediction
    prediction = model.predict(features)[0]

    # Probability (Optional)
    probability = model.predict_proba(features)[0]

    if prediction == 1:
        result = "🎉 Passenger Survived"
        confidence = probability[1] * 100
    else:
        result = "❌ Passenger Did Not Survive"
        confidence = probability[0] * 100

    return render_template(
        "index.html",
        prediction_text=result,
        confidence=round(confidence, 2)
    )


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)