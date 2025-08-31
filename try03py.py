# =============================================================================
# 1. IMPORT LIBRARIES
# =============================================================================
import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# =============================================================================
# 2. LOAD ARTIFACTS (MODEL, SCALER, FEATURE NAMES, AND DATA)
# =============================================================================
try:
    model = joblib.load("xgb_obesity_model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_names = model.feature_names_in_
    # For global explanation, we need a sample of the training data.
    # Let's assume the original (unscaled) data is available in a CSV.
    # Make sure 'obesity_dataset_original.csv' is in your repo.
    df_original = pd.read_csv("obesity_dataset_original.csv")
    X_original = df_original[feature_names]

except FileNotFoundError:
    st.error(
        "Error: A required file was not found. "
        "Please ensure 'xgb_obesity_model.pkl', 'scaler.pkl', and 'obesity_dataset_original.csv' "
        "are in your GitHub repository."
    )
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading files: {e}")
    st.stop()

# =============================================================================
# 3. GLOBAL VARIABLES / CONSTANTS
# =============================================================================
numeric_cols = ["Age", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]
label_mapping = {
    0: "Insufficient_Weight", 1: "Normal_Weight", 2: "Overweight_Level_I",
    3: "Overweight_Level_II", 4: "Obesity_Type_I", 5: "Obesity_Type_II",
    6: "Obesity_Type_III"
}

# =============================================================================
# 4. PAGE CONFIGURATION AND SESSION STATE INITIALIZATION
# =============================================================================
st.set_page_config(page_title="Obesity Risk Predictor", page_icon="🍔", layout="wide") # Use wide layout

if 'prediction_code' not in st.session_state:
    st.session_state.prediction_code = None
if 'prediction_label' not in st.session_state:
    st.session_state.prediction_label = None
if 'explanation_text' not in st.session_state:
    st.session_state.explanation_text = None
if 'df_input_original' not in st.session_state:
    st.session_state.df_input_original = None

# =============================================================================
# 5. UI: STYLING, TITLES, AND INPUT WIDGETS
# =============================================================================
st.markdown("""<style>...your css...</style>""", unsafe_allow_html=True)
st.markdown('<div class="title-font">Obesity Risk Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="cute-subtitle">A personalized obesity risk assessment with model explanations.</div>', unsafe_allow_html=True)

# Use columns for a cleaner layout
main_col, explanation_col = st.columns([1, 1])

with main_col:
    st.markdown('<div class="section-title">📝 Input Your Health Data</div>', unsafe_allow_html=True)
    # Input widgets
    Age = st.number_input("Age (years)", min_value=14, max_value=100, value=25, step=1)
    Weight = st.number_input("Weight (kg)", min_value=30.0, max_value=180.0, value=70.0, step=0.5)
    # ... (rest of your input widgets) ...
    MTRANS = st.selectbox("Primary transportation method", ["Public_Transportation", "Automobile", "Walking", "Motorbike", "Bike"])

# =============================================================================
# 6. PREDICTION LOGIC
# =============================================================================
if main_col.button("🔍 Predict Obesity Risk", use_container_width=True):
    # ... (Your prediction logic here, same as before) ...
    pass # This logic is unchanged

# =============================================================================
# 7. DISPLAY RESULTS AND LOCAL (TEXT) EXPLANATION
# =============================================================================
with explanation_col:
    st.markdown('<div class="section-title">💡 Prediction & Explanation</div>', unsafe_allow_html=True)
    if st.session_state.prediction_code is not None:
        st.success(f"✅ Your predicted obesity category is: **{st.session_state.prediction_label}**")

        if st.button("📊 Explain My Prediction", use_container_width=True):
             # ... (Your text explanation logic here, same as before) ...
             pass # This logic is unchanged
        
        if st.session_state.explanation_text:
            st.info(st.session_state.explanation_text)
    else:
        st.info("Please input your data and click 'Predict' to see the results.")

# =============================================================================
# 8. GLOBAL EXPLANATION (NEW SECTION)
# =============================================================================

# This function generates the global SHAP summary plot.
# The @st.cache_data decorator ensures this heavy computation runs only once.
@st.cache_data
def generate_global_shap_plot(_model, _data_sample):
    """
    Generates and returns a Matplotlib figure of the global SHAP summary plot.
    """
    # Scale the data sample just like the training data
    data_scaled = _data_sample.copy()
    data_scaled[numeric_cols] = scaler.transform(data_scaled[numeric_cols])

    # Create the explainer and calculate SHAP values for the sample
    explainer = shap.Explainer(_model)
    shap_values = explainer.shap_values(data_scaled)
    
    # Create the plot
    fig, ax = plt.subplots()
    class_names_list = [label_mapping[i] for i in sorted(label_mapping.keys())]
    shap.summary_plot(
        shap_values, 
        data_scaled, 
        plot_type="bar", 
        class_names=class_names_list,
        show=False
    )
    plt.title("SHAP Summary Plot: Global Feature Importance")
    plt.tight_layout()
    return fig

# Create an expander to show the global plot
with st.expander("🔬 Show Model's Overall Feature Importance (Global Explanation)"):
    st.markdown("""
    This chart shows which features the model considers most important **overall**, across all predictions. 
    It helps us understand the model's general behavior.
    """)
    # To make it faster, we use a sample of the data for the global plot.
    # Using 500 samples is usually enough to get a good representation.
    fig = generate_global_shap_plot(model, X_original.sample(500, random_state=42))
    st.pyplot(fig)

# =============================================================================
# 9. FOOTER
# =============================================================================
st.markdown('<div class="footer">Made with ❤️ by Your AI Assistant</div>', unsafe_allow_html=True)