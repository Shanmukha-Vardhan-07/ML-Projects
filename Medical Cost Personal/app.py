print("MEDICAL INSURANCE APP")

from flask import Flask, render_template, request
import pickle
import pandas as pd


app = Flask(__name__)


# Load files
model = pickle.load(open("insurance_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # Get input values
    age = int(request.form["age"])
    sex = request.form["sex"]
    bmi = float(request.form["bmi"])
    children = int(request.form["children"])
    smoker = request.form["smoker"]
    region = request.form["region"]


    # Create dataframe
    input_data = pd.DataFrame({
        "age": [age],
        "sex": [sex],
        "bmi": [bmi],
        "children": [children],
        "smoker": [smoker],
        "region": [region]
    })


    # Same encoding as training
    input_data = pd.get_dummies(
        input_data,
        drop_first=True
    )


    # Match training columns
    input_data = input_data.reindex(
        columns=columns,
        fill_value=0
    )


    # Scaling - ONLY the numeric columns the scaler was actually fit on
    numeric_cols = ["age", "bmi", "children"]
    input_data[numeric_cols] = scaler.transform(input_data[numeric_cols])


    # Prediction (full 8-column row: scaled numeric + untouched dummy columns)
    prediction = model.predict(input_data)


    result = round(prediction[0], 2)


    return render_template(
        "index.html",
        prediction=f"Estimated Insurance Cost: ₹ {result}"
    )


if __name__ == "__main__":
    app.run(debug=True)