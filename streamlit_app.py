import streamlit as st
import pickle
import numpy as np

# ================= LOAD FILES =================
try:
    model = pickle.load(open("model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    feature_columns = pickle.load(open("features.pkl", "rb"))
except Exception as e:
    st.error(f"❌ Error loading model files: {e}")
    st.stop()

# ================= UI =================
st.set_page_config(page_title="Churn Prediction", layout="wide")

st.title("📊 Customer Churn Prediction")
st.markdown("Predict whether a customer will churn or not")

st.sidebar.header("📥 Enter Customer Details")

# ================= INPUTS =================
with st.sidebar.expander("Demographics"):
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    gender = st.selectbox("Gender", ["Female", "Male"])
    partner = st.selectbox("Partner", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])

with st.sidebar.expander("Services"):
    phone = st.selectbox("Phone Service", ["No", "Yes"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

with st.sidebar.expander("Billing"):
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["No", "Yes"])
    payment = st.selectbox("Payment Method", [
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check"
    ])

with st.sidebar.expander("Charges"):
    tenure = st.slider("Tenure (months)", 0, 72)
    monthly_charges = st.slider("Monthly Charges", 0.0, 150.0)

# ================= FEATURE ENGINEERING =================
input_data = dict.fromkeys(feature_columns, 0)

# Numerical
input_data["tenure"] = tenure
input_data["MonthlyCharges"] = monthly_charges
input_data["TotalCharges"] = tenure * monthly_charges  # approx

# Binary
input_data["SeniorCitizen"] = 1 if senior == "Yes" else 0
if gender == "Male": input_data["gender_Male"] = 1
if partner == "Yes": input_data["Partner_Yes"] = 1
if dependents == "Yes": input_data["Dependents_Yes"] = 1
if phone == "Yes": input_data["PhoneService_Yes"] = 1

# MultipleLines
if multiple_lines == "Yes":
    input_data["MultipleLines_Yes"] = 1
elif multiple_lines == "No phone service":
    input_data["MultipleLines_No phone service"] = 1

# Internet
if internet == "Fiber optic":
    input_data["InternetService_Fiber optic"] = 1
elif internet == "No":
    input_data["InternetService_No"] = 1

# Contract
if contract == "One year":
    input_data["Contract_One year"] = 1
elif contract == "Two year":
    input_data["Contract_Two year"] = 1

# Paperless
if paperless == "Yes": input_data["PaperlessBilling_Yes"] = 1

# Payment
if payment == "Credit card (automatic)":
    input_data["PaymentMethod_Credit card (automatic)"] = 1
elif payment == "Electronic check":
    input_data["PaymentMethod_Electronic check"] = 1
elif payment == "Mailed check":
    input_data["PaymentMethod_Mailed check"] = 1

# Charge Range
if 35 <= monthly_charges <= 70:
    input_data["ChargeRange_Mid (35–70)"] = 1
elif 70 < monthly_charges <= 120:
    input_data["ChargeRange_High (70–120)"] = 1

# ================= PREDICTION =================
st.markdown("## 📈 Prediction Result")

if st.button("Predict"):
    try:
        features = np.array([list(input_data.values())])
        features = scaler.transform(features)
        prediction = model.predict(features)[0]

        # Probability if supported
        probability = None
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(features)[0][1]

        if prediction == 1:
            st.error("⚠️ Customer is likely to churn")
        else:
            st.success("✅ Customer will stay")

        if probability is not None:
            st.metric("Churn Probability", f"{round(probability * 100, 2)}%")

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")

# ================= FOOTER =================
st.markdown("---")
st.caption("Built with Streamlit | Telecom Churn Prediction Project")
