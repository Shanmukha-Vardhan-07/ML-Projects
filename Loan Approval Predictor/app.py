from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model, scaler, and label encoder
# Place loan_model.pkl, scaler.pkl, and label_encoder.pkl in the same
# folder as this app.py file
model = joblib.load("loan_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# This MUST match the exact column order produced by
# pd.get_dummies(X_train, drop_first=True) in the training notebook:
# ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
#  'Credit_History', 'Gender_Male', 'Married_Yes', 'Dependents_1',
#  'Dependents_2', 'Dependents_3+', 'Education_Not Graduate',
#  'Self_Employed_Yes', 'Property_Area_Semiurban', 'Property_Area_Urban']

def build_feature_vector(form):
    applicant_income = float(form.get("ApplicantIncome", 0))
    coapplicant_income = float(form.get("CoapplicantIncome", 0))
    loan_amount = float(form.get("LoanAmount", 0))
    loan_amount_term = float(form.get("Loan_Amount_Term", 360))
    credit_history = float(form.get("Credit_History", 1))

    gender = form.get("Gender", "Male")
    married = form.get("Married", "No")
    dependents = form.get("Dependents", "0")
    education = form.get("Education", "Graduate")
    self_employed = form.get("Self_Employed", "No")
    property_area = form.get("Property_Area", "Urban")

    # One-hot encode manually (same logic as pd.get_dummies(drop_first=True))
    gender_male = 1 if gender == "Male" else 0
    married_yes = 1 if married == "Yes" else 0

    dependents_1 = 1 if dependents == "1" else 0
    dependents_2 = 1 if dependents == "2" else 0
    dependents_3plus = 1 if dependents == "3+" else 0

    education_not_graduate = 1 if education == "Not Graduate" else 0
    self_employed_yes = 1 if self_employed == "Yes" else 0

    property_semiurban = 1 if property_area == "Semiurban" else 0
    property_urban = 1 if property_area == "Urban" else 0

    features = [
        applicant_income,
        coapplicant_income,
        loan_amount,
        loan_amount_term,
        credit_history,
        gender_male,
        married_yes,
        dependents_1,
        dependents_2,
        dependents_3plus,
        education_not_graduate,
        self_employed_yes,
        property_semiurban,
        property_urban,
    ]
    return np.array(features).reshape(1, -1)


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", result=None)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        X = build_feature_vector(request.form)
        X_scaled = scaler.transform(X)
        pred = model.predict(X_scaled)
        label = label_encoder.inverse_transform(pred)[0]  # 'Y' or 'N'

        proba = None
        if hasattr(model, "predict_proba"):
            proba = round(float(np.max(model.predict_proba(X_scaled))) * 100, 2)

        if label == "Y":
            result = {
                "status": "approved",
                "message": "Congratulations! Your loan is likely to be approved.",
                "confidence": proba,
            }
        else:
            result = {
                "status": "rejected",
                "message": "Unfortunately, your loan is likely to be rejected.",
                "confidence": proba,
            }

        return render_template("index.html", result=result, form_data=request.form)

    except Exception as e:
        error_result = {"status": "error", "message": f"Error: {str(e)}", "confidence": None}
        return render_template("index.html", result=error_result, form_data=request.form)


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)