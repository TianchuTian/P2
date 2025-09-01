# =============================================================================
# FILE 1: try04.py
# This is the main page for user data input.
# =============================================================================
import streamlit as st
import pandas as pd
import joblib

# Set page configuration FIRST. This must be the first Streamlit command.
st.set_page_config(page_title="Obesity Risk Predictor - Home", page_icon="🏠", layout="centered")

# --- Function to render the main input page ---
def main_page():
    """
    This function contains all the UI elements and logic for the input page.
    """
    # --- Load feature names for structuring the input ---
    try:
        model = joblib.load("xgb_obesity_model.pkl")
        feature_names = model.feature_names_in_
    except Exception as e:
        st.error(f"Error loading necessary files: {e}")
        st.stop()
    
# --- UI: Styling and Titles ---
st.markdown("""<style>
.title-font { font-size: 48px; font-weight: bold; color: #2c3e50; text-align: center; font-family: 'Arial Rounded MT Bold', sans-serif; }
.cute-subtitle { font-size: 22px; font-family: "Comic Sans MS", cursive, sans-serif; color: #e17055; text-align: center; margin-bottom: 30px; }
.section-title { font-size: 24px; font-weight: bold; color: #0984e3; margin-top: 40px; margin-bottom: 20px; }
.footer { margin-top: 60px; text-align: center; font-size: 14px; color: #636e72; }
</style>""", unsafe_allow_html=True)
st.markdown('<div class="title-font">Obesity Risk Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="cute-subtitle">A personalized obesity risk assessment with AI-powered explanations.</div>', unsafe_allow_html=True)

# --- Main Input Form ---
# Using st.form ensures that all inputs are collected together when the button is pressed.
with st.form("user_input_form"):
    st.markdown('<div class="section-title">📝 Input Your Health Data</div>', unsafe_allow_html=True)
    
    # Use columns for a better layout within the form
    col1, col2 = st.columns(2)
    with col1:
        Age = st.number_input("Age (years)", 14, 100, 25, 1)
        Weight = st.number_input("Weight (kg)", 30.0, 180.0, 70.0, 0.5)
        FCVC = st.slider("[FCVC] Frequency of vegetable consumption (1=low, 3=high)", 1.0, 3.0, 1.0, 0.1)
        NCP = st.slider("[NCP] Number of main meals per day (1-4)", 1.0, 4.0, 4.0, 0.1)
        CH2O = st.slider("[CH2O] Daily water intake (1=low, 3=high)", 1.0, 3.0, 1.0, 0.1)
        FAF = st.slider("[FAF] Physical activity frequency per week (0=none, 3=frequent)", 0.0, 3.0, 0.0, 0.1)
        TUE = st.slider("[TUE] Daily screen time (0=low, 2=high)", 0.0, 2.0, 2.0, 0.1)
    
    with col2:
        Gender = st.selectbox("Gender", ["Male", "Female"])
        family_history_with_overweight = st.selectbox("Family history of overweight", ["yes", "no"])
        FAVC = st.selectbox("[FAVC] Do you consume high caloric food frequently?", ["yes", "no"], index=0)
        CAEC = st.selectbox("[CAEC] Snacking frequency", ["no", "Sometimes", "Frequently", "Always"], index=3)
        CALC = st.selectbox("[CALC] Alcohol consumption", ["no", "Sometimes", "Frequently", "Always"], index=3)
        MTRANS = st.selectbox("[MTRANS] Primary transportation method", ["Public_Transportation", "Automobile", "Walking", "Motorbike", "Bike"], index=1)

    # Form submission button
    submitted = st.form_submit_button("🔍 Predict & View My Report")

    if submitted:
        # Create input dictionary from the form's state
        input_dict = {
            "Gender": 1 if Gender == "Male" else 0, "Age": Age, "Weight": Weight,
            "family_history_with_overweight": 1 if family_history_with_overweight == "yes" else 0,
            "FAVC": 1 if FAVC == "yes" else 0, "FCVC": FCVC, "NCP": NCP, "CH2O": CH2O,
            "FAF": FAF, "TUE": TUE,
            "CAEC_Always": 1 if CAEC == "Always" else 0, "CAEC_Frequently": 1 if CAEC == "Frequently" else 0,
            "CAEC_Sometimes": 1 if CAEC == "Sometimes" else 0, "CAEC_no": 1 if CAEC == "no" else 0,
            "CALC_Always": 1 if CALC == "Always" else 0, "CALC_Frequently": 1 if CALC == "Frequently" else 0,
            "CALC_Sometimes": 1 if CALC == "Sometimes" else 0, "CALC_no": 1 if CALC == "no" else 0,
            "MTRANS_Automobile": 1 if MTRANS == "Automobile" else 0, "MTRANS_Bike": 1 if MTRANS == "Bike" else 0,
            "MTRANS_Motorbike": 1 if MTRANS == "Motorbike" else 0,
            "MTRANS_Public_Transportation": 1 if MTRANS == "Public_Transportation" else 0,
            "MTRANS_Walking": 1 if MTRANS == "Walking" else 0,
        }
        
        # Store the complete input dictionary in session state to pass it to the report page
        st.session_state.user_input = input_dict
        
        # Instead of calling switch_page directly, set the navigation flag to True.
        # The app will rerun, and the logic at the top will handle the page switch.
        st.session_state.navigate_to_report = True

st.markdown('<div class="footer">Made with ❤️ by Your AI Assistant</div>', unsafe_allow_html=True)

# =============================================================================
# SCRIPT EXECUTION LOGIC
# =============================================================================
# This is the main router of the app.
if st.session_state.get("navigate_to_report", False):
    # If the navigation flag is set, switch to the report page.
    st.session_state.navigate_to_report = False
    st.switch_page("pages/1_📊_Report.py")
else:
    # Otherwise, render the main input page.
    main_page()
