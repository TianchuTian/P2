
import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Page configuration
st.set_page_config(page_title="Obesity Risk Predictor", page_icon="🍔", layout="centered")

# Custom CSS
st.markdown("""
<style>
.title-font {
    font-size: 48px;
    font-weight: bold;
    color: #2c3e50;
    text-align: center;
    font-family: 'Arial Rounded MT Bold', sans-serif;
}
.cute-subtitle {
    font-size: 22px;
    font-family: "Comic Sans MS", cursive, sans-serif;
    color: #e17055;
    text-align: center;
    margin-bottom: 30px;
}
.section-title {
    font-size: 24px;
    font-weight: bold;
    color: #0984e3;
    margin-top: 40px;
}
.footer {
    margin-top: 60px;
    text-align: center;
    font-size: 14px;
    color: #636e72;
}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="title-font">Obesity Risk Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="cute-subtitle">Get a personalized obesity risk assessment based on your health habits.</div>', unsafe_allow_html=True)

# Input Section
st.markdown('<div class="section-title">📝 Input Your Health Data</div>', unsafe_allow_html=True)

# Validated number input
def validated_number(label, min_val, max_val, step=1.0, is_int=False):
    value = st.number_input(
        f"{label} ({min_val} - {max_val})",
        min_value=int(min_val) if is_int else float(min_val),
        max_value=int(max_val) if is_int else float(max_val),
        value=int(min_val) if is_int else float(min_val),
        step=1 if is_int else step
    )
    return int(value) if is_int else float(value)

# Numerical Inputs
Age = validated_number("Age (years)", 14, 60, is_int=True)
Weight = validated_number("Weight (kg)", 40, 150, is_int=True)
FCVC = validated_number("Frequency of vegetable consumption (1=low, 3=high)", 1.0, 3.0, 0.1)
NCP = validated_number("Number of main meals (1=low, 4=high)", 1.0, 4.0, 0.1)
CH2O = validated_number("Water intake frequency (1=low, 3=frequent)", 1.0, 3.0, 0.1)
FAF = validated_number("Physical activity frequency (0=none, 3=frequent)", 0.0, 3.0, 0.1)
TUE = validated_number("Daily screen time (0=none, 2=high)", 0.0, 2.0, 0.1)

# Categorical Inputs
Gender = st.selectbox("Gender", ["Male", "Female"])
family_history_with_overweight = st.selectbox("Family history of overweight", ["yes", "no"])
FAVC = st.selectbox("Do you consume high caloric food frequently?", ["yes", "no"])
SMOKE = st.selectbox("Do you smoke?", ["yes", "no"])
CAEC = st.selectbox("Snacking frequency", ["no", "Sometimes", "Frequently", "Always"])
CALC = st.selectbox("Alcohol consumption", ["no", "Sometimes", "Frequently", "Always"])
MTRANS = st.selectbox("Transportation method", ["Public_Transportation", "Automobile", "Walking", "Motorbike", "Bike"])

# Prediction logic
if st.button("🔍 Predict Obesity Risk"):
    input_dict = {
        "Gender": 1 if Gender == "Male" else 0,
        "Age": Age,
        "Weight": Weight,
        "family_history_with_overweight": 1 if family_history_with_overweight == "yes" else 0,
        "FAVC": 1 if FAVC == "yes" else 0,
        "FCVC": FCVC,
        "NCP": NCP,
        "CH2O": CH2O,
        "FAF": FAF,
        "TUE": TUE,
        "SMOKE": 1 if SMOKE == "yes" else 0,
        "CAEC_Always": 1 if CAEC == "Always" else 0,
        "CAEC_Frequently": 1 if CAEC == "Frequently" else 0,
        "CAEC_Sometimes": 1 if CAEC == "Sometimes" else 0,
        "CAEC_no": 1 if CAEC == "no" else 0,
        "CALC_Always": 1 if CALC == "Always" else 0,
        "CALC_Frequently": 1 if CALC == "Frequently" else 0,
        "CALC_Sometimes": 1 if CALC == "Sometimes" else 0,
        "CALC_no": 1 if CALC == "no" else 0,
        "MTRANS_Automobile": 1 if MTRANS == "Automobile" else 0,
        "MTRANS_Bike": 1 if MTRANS == "Bike" else 0,
        "MTRANS_Motorbike": 1 if MTRANS == "Motorbike" else 0,
        "MTRANS_Public_Transportation": 1 if MTRANS == "Public_Transportation" else 0,
        "MTRANS_Walking": 1 if MTRANS == "Walking" else 0,
    }

    df_input = pd.DataFrame([input_dict])

    # Normalize numerical columns
    numeric_cols = ["Age", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]
    df_input[numeric_cols] = scaler.transform(df_input[numeric_cols])

    # Predict
    prediction = model.predict(df_input)[0]

    st.success(f"✅ Your predicted obesity category is: **{prediction}**")

# Footer
st.markdown('<div class="footer">Made with ❤️ by Your AI Assistant</div>', unsafe_allow_html=True)
