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
            # --- UPGRADED FEATURE TRANSLATOR for more narrative phrases ---
            feature_translator = {
                'Weight': lambda v: f"a weight of **{v:.0f} kg**",
                'Age': lambda v: f"an age of **{v:.0f} years**",
                'FCVC': lambda v: "a **low** frequency of vegetable consumption" if v < 1.5 else ("a **moderate** vegetable consumption" if v < 2.5 else "a **high** vegetable consumption"),
                'FAF': lambda v: "a **low** frequency of physical activity" if v < 1.5 else ("a **moderate** amount of physical activity" if v < 2.5 else "a **high** amount of physical activity"),
                'family_history_with_overweight': lambda v: "a **family history of overweight**" if v == 1 else None,
                'FAVC': lambda v: "the **frequent consumption of high-caloric food**" if v == 1 else None,
                'NCP': lambda v: "a **low** number of main meals" if v < 2.5 else "a **normal** number of main meals",
                'CH2O': lambda v: "**low** daily water intake" if v < 1.5 else "**adequate** daily water intake",
                'TUE': lambda v: "**short** daily screen time" if v < 1.0 else "**long** daily screen time",
                'CAEC_Sometimes': lambda v: "a habit of **sometimes snacking**" if v == 1 else None,
                'CAEC_Frequently': lambda v: "a habit of **frequently snacking**" if v == 1 else None,
            }

            with st.spinner('Analyzing the reasons for your prediction...'):
                try:
                    df_scaled = pd.DataFrame(st.session_state.df_input_original, columns=feature_names)
                    df_scaled[numeric_cols] = scaler.transform(df_scaled[numeric_cols])
                    
                    explainer = shap.Explainer(model)
                    shap_values = explainer.shap_values(df_scaled)
                    
                    predicted_class_index = st.session_state.prediction_code
                    
                    shap_df = pd.DataFrame(
                        shap_values[0, :, predicted_class_index],
                        index=feature_names,
                        columns=['shap_value']
                    )
                    shap_df['feature_value'] = st.session_state.df_input_original.iloc[0].values
                    
                    risk_factors_df = shap_df[shap_df['shap_value'] > 0.01].sort_values('shap_value', ascending=False)
                    protective_factors_df = shap_df[shap_df['shap_value'] < -0.01].sort_values('shap_value', ascending=True)

                    # --- NEW NARRATIVE GENERATION LOGIC ---
                    pred_label = st.session_state.prediction_label
                    
                    # 1. Opening statement
                    explanation = f"Based on the information you provided, the model predicted your risk category as **{pred_label}**.\n\n"
                    
                    # 2. Identify and describe the primary driver
                    if not risk_factors_df.empty:
                        primary_driver_name = risk_factors_df.index[0]
                        primary_driver_value = risk_factors_df.iloc[0]['feature_value']
                        
                        # Try to find a secondary driver for more context
                        secondary_driver_name = risk_factors_df.index[1] if len(risk_factors_df) > 1 else None
                        
                        if primary_driver_name in ['Weight', 'Age'] and secondary_driver_name in ['Weight', 'Age']:
                            # Create the contextual opening like "For your age, your weight is..."
                            age_val = st.session_state.df_input_original['Age'].iloc[0]
                            weight_val = st.session_state.df_input_original['Weight'].iloc[0]
                            explanation += f"For an individual of **{age_val:.0f} years old**, a weight of **{weight_val:.0f} kg** was the most significant factor leading to the '{pred_label}' prediction. "
                        else:
                            # Generic but still strong opening
                            primary_driver_text = feature_translator.get(primary_driver_name, lambda v: primary_driver_name)(primary_driver_value)
                            explanation += f"The most significant factor leading to this prediction was **{primary_driver_text}**. "
                    
                    # 3. List other contributing factors
                    other_risk_factors = risk_factors_df.iloc[1:] # Skip the primary one if it was already used
                    if not other_risk_factors.empty:
                        explanation += "Other contributing risk factors include:\n"
                        for feature, row in other_risk_factors.head(2).iterrows(): # List up to 2 more
                             if feature in feature_translator:
                                text = feature_translator[feature](row['feature_value'])
                                if text:
                                    explanation += f"- {text.capitalize()}\n"

                    # 4. Mention protective factors
                    if not protective_factors_df.empty:
                        explanation += "\nOn the other hand, some habits are helping to lower your risk. The main protective factor is:\n"
                        for feature, row in protective_factors_df.head(1).iterrows():
                            if feature in feature_translator:
                                text = feature_translator[feature](row['feature_value'])
                                if text:
                                    explanation += f"- {text.capitalize()}\n"
                                    
                    st.session_state.explanation_text = explanation

                except Exception as e:
                    st.error(f"Sorry, the explanation could not be generated: {e}")
        
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
    fig = generate_global_shap_plot(model, X_original.sample(500, random_state=42))
    st.pyplot(fig)
    

# =============================================================================
# 9. FOOTER
# =============================================================================
st.markdown('<div class="footer">Made with ❤️ by Your AI Assistant</div>', unsafe_allow_html=True)
