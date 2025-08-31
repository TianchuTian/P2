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
st.markdown("""<style>...your css...</style>""", unsafe_allow_html=True) # Keeping this short
st.markdown('<div class="title-font">Obesity Risk Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="cute-subtitle">A personalized obesity risk assessment with model explanations.</div>', unsafe_allow_html=True)

# Use columns for a cleaner layout
main_col, explanation_col = st.columns([1, 1])

with main_col:
    st.markdown('<div class="section-title">📝 Input Your Health Data</div>', unsafe_allow_html=True)
    
    # Input widgets - THIS IS THE FULL, CORRECTED LIST
    Age = st.number_input("Age (years)", min_value=14, max_value=100, value=25, step=1)
    Weight = st.number_input("Weight (kg)", min_value=30.0, max_value=180.0, value=70.0, step=0.5)
    FCVC = st.slider("Frequency of vegetable consumption (1=low, 3=high)", 1.0, 3.0, 2.0, 0.1)
    NCP = st.slider("Number of main meals per day (1-4)", 1.0, 4.0, 3.0, 0.1)
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
# 6. PREDICTION LOGIC
# =============================================================================
# --- Prediction Button and its Logic ---
    if st.button("🔍 Predict Obesity Risk", use_container_width=True):
        # Create input dictionary from UI widgets.
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
        # Create DataFrame and ensure correct column order.
        df_input = pd.DataFrame([input_dict])[feature_names]
        
        # Scale the data for the model.
        df_input_scaled = df_input.copy()
        df_input_scaled[numeric_cols] = scaler.transform(df_input_scaled[numeric_cols])
        
        # Make the prediction.
        prediction_code = model.predict(df_input_scaled)[0]
        prediction_label = label_mapping.get(prediction_code, "Unknown")
        
        # Save results to the session state to persist them across reruns.
        st.session_state.prediction_code = prediction_code
        st.session_state.prediction_label = prediction_label
        st.session_state.df_input_original = df_input
        st.session_state.explanation_text = None # Reset explanation for new prediction.


# =============================================================================
# 7. DISPLAY RESULTS AND LOCAL (TEXT) EXPLANATION
# =============================================================================
with explanation_col:
    st.markdown('<div class="section-title">💡 Prediction & Explanation</div>', unsafe_allow_html=True)
    if st.session_state.prediction_code is not None:
        st.success(f"✅ Your predicted obesity category is: **{st.session_state.prediction_label}**")

        if st.button("📊 Explain My Prediction", use_container_width=True):
            # The translator dictionary to convert feature values to human-readable text.
            feature_translator = {
                'Weight': {'name': 'Your weight', 'map': lambda v: f"({v:.0f} kg)"},
                'Age': {'name': 'Your age', 'map': lambda v: f"({v:.0f} years old)"},
                'FCVC': {'name': 'Your vegetable consumption', 'map': lambda v: "is low" if v < 1.5 else ("is moderate" if v < 2.5 else "is high")},
                'FAF': {'name': 'Your physical activity frequency', 'map': lambda v: "is low" if v < 1.5 else ("is moderate" if v < 2.5 else "is high")},
                'family_history_with_overweight': {'name': 'A family history of overweight', 'map': lambda v: "" if v == 1 else None},
                'FAVC': {'name': 'Frequent consumption of high-caloric food', 'map': lambda v: "" if v == 1 else None},
                'NCP': {'name': 'Your number of main meals', 'map': lambda v: "is low" if v < 2.5 else "is normal"},
                'CH2O': {'name': 'Your daily water intake', 'map': lambda v: "is low" if v < 1.5 else "is adequate"},
                'TUE': {'name': 'Your screen time', 'map': lambda v: "is short" if v < 1.0 else "is long"},
                'CAEC_Sometimes': {'name': 'Your snacking habit', 'map': lambda v: "is 'Sometimes'" if v == 1 else None},
                'CAEC_Frequently': {'name': 'Your snacking habit', 'map': lambda v: "is 'Frequently'" if v == 1 else None},
                'CAEC_Always': {'name': 'Your snacking habit', 'map': lambda v: "is 'Always'" if v == 1 else None},
                'MTRANS_Automobile': {'name': 'Your primary transport', 'map': lambda v: "is Automobile" if v == 1 else None},
                'MTRANS_Public_Transportation': {'name': 'Your primary transport', 'map': lambda v: "is Public Transportation" if v == 1 else None},
            }

            with st.spinner('Analyzing the reasons for your prediction...'):
                try:
                    # Prepare scaled data for SHAP.
                    df_scaled = pd.DataFrame(st.session_state.df_input_original, columns=feature_names)
                    df_scaled[numeric_cols] = scaler.transform(df_scaled[numeric_cols])

                    explainer = shap.Explainer(model)
                    shap_values = explainer.shap_values(df_scaled)
                    
                    predicted_class_index = st.session_state.prediction_code
                    
                    # Create a DataFrame to analyze SHAP values and original feature values.
                    shap_df = pd.DataFrame(
                        shap_values[0, :, predicted_class_index], # Slice correctly for the predicted class
                        index=feature_names,
                        columns=['shap_value']
                    )
                    shap_df['feature_value'] = st.session_state.df_input_original.iloc[0].values
                    
                    # Identify the top risk and protective factors.
                    risk_factors = shap_df[shap_df['shap_value'] > 0.01].sort_values('shap_value', ascending=False).head(3)
                    protective_factors = shap_df[shap_df['shap_value'] < -0.01].sort_values('shap_value', ascending=True).head(1)
                    
                    # Build the explanation text dynamically.
                    explanation = f"The model predicted your risk category as **{st.session_state.prediction_label}**.\n\n"
                    
                    if not risk_factors.empty:
                        explanation += "The main **risk factors** contributing to this prediction are:\n"
                        for feature, row in risk_factors.iterrows():
                            if feature in feature_translator:
                                translation = feature_translator[feature]
                                value_text = translation['map'](row['feature_value'])
                                if value_text is not None:
                                    explanation += f"- **{translation['name']}** {value_text}\n"

                    if not protective_factors.empty:
                        explanation += "\nOn the other hand, the main **protective factor** is:\n"
                        for feature, row in protective_factors.iterrows():
                            if feature in feature_translator:
                                translation = feature_translator[feature]
                                value_text = translation['map'](row['feature_value'])
                                if value_text is not None:
                                    explanation += f"- **{translation['name']}** {value_text}\n"

                    st.session_state.explanation_text = explanation

                except Exception as e:
                    st.error(f"Sorry, the explanation could not be generated: {e}")
        
        # Display the text if it has been generated.
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
    data_scaled = _data_sample.copy()
    data_scaled[numeric_cols] = scaler.transform(data_scaled[numeric_cols])

    explainer = shap.Explainer(_model)
    shap_values = explainer.shap_values(data_scaled)
    
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

with st.expander("🔬 Show Model's Overall Feature Importance (Global Explanation)"):
    st.markdown("""
    This chart shows which features the model considers most important **overall**, across all predictions. 
    It helps us understand the model's general behavior.
    """)
    # To make it faster, we use a sample of the data for the global plot.
    fig = generate_global_shap_plot(model, X_original_for_sampling.sample(500, random_state=42))
    st.pyplot(fig)
    

# =============================================================================
# 9. FOOTER
# =============================================================================
st.markdown('<div class="footer">Made with ❤️ by Your AI Assistant</div>', unsafe_allow_html=True)
