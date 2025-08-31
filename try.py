# =============================================================================
# 1. IMPORT LIBRARIES
# =============================================================================
import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# =============================================================================
# 2. LOAD ARTIFACTS (MODEL, SCALER, FEATURE NAMES)
# =============================================================================
try:
    model = joblib.load("xgb_obesity_model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_names = joblib.load("feature_names.pkl")
except FileNotFoundError:
    st.error(
        "Error: Model or helper files not found. "
        "Please make sure 'xgb_obesity_model.pkl', 'scaler.pkl', and 'feature_names.pkl' "
        "are in your GitHub repository."
    )
    st.stop()

# =============================================================================
# 3. PAGE CONFIGURATION AND SESSION STATE INITIALIZATION
# =============================================================================
st.set_page_config(page_title="Obesity Risk Predictor", page_icon="🍔", layout="centered")

# Initialize all session state variables at the start.
if 'prediction_code' not in st.session_state:
    st.session_state.prediction_code = None
if 'prediction_label' not in st.session_state:
    st.session_state.prediction_label = None
if 'shap_plot' not in st.session_state:
    st.session_state.shap_plot = None
if 'df_input_for_shap' not in st.session_state:
    st.session_state.df_input_for_shap = None

# =============================================================================
# 4. UI: STYLING, TITLES, AND INPUT WIDGETS
# =============================================================================
st.markdown("""<style>...your css...</style>""", unsafe_allow_html=True) # Keeping this short
st.markdown('<div class="title-font">Obesity Risk Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="cute-subtitle">Get a personalized obesity risk assessment based on your health habits.</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">📝 Input Your Health Data</div>', unsafe_allow_html=True)

# Input Widgets
col1, col2 = st.columns(2)
with col1:
    Age = st.number_input("Age (years)", min_value=14, max_value=100, value=25, step=1)
    Weight = st.number_input("Weight (kg)", min_value=30.0, max_value=180.0, value=70.0, step=0.5)
    FCVC = st.slider("Frequency of vegetable consumption (1=low, 3=high)", 1.0, 3.0, 2.0, 0.1)
    NCP = st.slider("Number of main meals per day (1-4)", 1.0, 4.0, 3.0, 0.1)
with col2:
    CH2O = st.slider("Daily water intake (1=low, 3=high)", 1.0, 3.0, 2.0, 0.1)
    FAF = st.slider("Physical activity frequency per week (0=none, 3=frequent)", 0.0, 3.0, 1.0, 0.1)
    TUE = st.slider("Daily screen time (0=low, 2=high)", 0.0, 2.0, 1.0, 0.1)

Gender = st.selectbox("Gender", ["Male", "Female"])
family_history_with_overweight = st.selectbox("Family history of overweight", ["yes", "no"])
FAVC = st.selectbox("Do you consume high caloric food frequently?", ["yes", "no"])
CAEC = st.selectbox("Snacking frequency", ["no", "Sometimes", "Frequently", "Always"])
CALC = st.selectbox("Alcohol consumption", ["no", "Sometimes", "Frequently", "Always"])
MTRANS = st.selectbox("Primary transportation method", ["Public_Transportation", "Automobile", "Walking", "Motorbike", "Bike"])

# =============================================================================
# 5. PREDICTION LOGIC
# =============================================================================
if st.button("🔍 Predict Obesity Risk"):
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

    df_input = pd.DataFrame([input_dict])[feature_names]
    df_input_scaled = df_input.copy()
    numeric_cols = ["Age", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]
    df_input_scaled[numeric_cols] = scaler.transform(df_input_scaled[numeric_cols])

    label_mapping = {
        0: "Insufficient_Weight", 1: "Normal_Weight", 2: "Overweight_Level_I",
        3: "Overweight_Level_II", 4: "Obesity_Type_I", 5: "Obesity_Type_II",
        6: "Obesity_Type_III"
    }
    
    prediction_code = model.predict(df_input_scaled)[0]
    prediction_label = label_mapping.get(prediction_code, "Unknown")
    
    st.session_state.prediction_code = prediction_code
    st.session_state.prediction_label = prediction_label
    st.session_state.df_input_for_shap = df_input_scaled
    st.session_state.shap_plot = None

# =============================================================================
# 6. DISPLAY RESULTS AND EXPLANATION LOGIC
# =============================================================================
# --- ROBUSTNESS FIX IS HERE ---
# We now check for 'prediction_code' instead of 'prediction_label'.
# This ensures the success message ONLY appears if we have a valid code to explain.
# The number 0 is a valid prediction code, so 'is not None' is the correct check.
if st.session_state.prediction_code is not None:
    st.success(f"✅ Your predicted obesity category is: **{st.session_state.prediction_label}**")

    if st.button("📊 Explain Prediction"):
        with st.spinner('Generating explanation, please wait...'):
            try:
                explainer = shap.Explainer(model)
                shap_values = explainer.shap_values(st.session_state.df_input_for_shap)
                
                predicted_class_index = st.session_state.prediction_code
                
                # Plotting the SHAP force plot
                shap.force_plot(
                    explainer.expected_value[predicted_class_index],
                    shap_values[predicted_class_index][0,:],
                    st.session_state.df_input_for_shap.iloc[0,:],
                    matplotlib=True,
                    show=False
                )
                
                st.session_state.shap_plot = plt.gcf()
                # Use bbox_inches to prevent labels from being cut off.
                plt.tight_layout()
                # No need to save to file unless for debugging.
                # plt.savefig("shap_plot.png", bbox_inches='tight') 
                plt.close() # Close the plot to free up memory

            except Exception as e:
                st.error(f"Sorry, the explanation could not be generated: {e}")

if st.session_state.shap_plot:
    st.markdown('<div class="section-title">💡 Prediction Explanation</div>', unsafe_allow_html=True)
    st.pyplot(st.session_state.shap_plot)
    st.markdown("""
    **How to read this chart:**
    - **Base value**: The average prediction output for this class.
    - **<span style="color:red;">Red arrows</span>**: Features pushing the prediction score **higher** (risk factors).
    - **<span style="color:blue;">Blue arrows</span>**: Features pushing the prediction score **lower** (protective factors).
    - The **length** of an arrow shows the magnitude of that feature's impact.
    """, unsafe_allow_html=True)

# =============================================================================
# 7. FOOTER
# =============================================================================
st.markdown('<div class="footer">Made with ❤️ by Your AI Assistant</div>', unsafe_allow_html=True)