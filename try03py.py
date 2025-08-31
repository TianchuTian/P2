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
except FileNotFoundError:
    st.error("Error: Model or scaler files not found. Please ensure all necessary files are in your GitHub repository.")
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
st.set_page_config(page_title="Obesity Risk Predictor", page_icon="🍔", layout="wide")

if 'prediction_label' not in st.session_state:
    st.session_state.prediction_label = None
if 'explanation_text' not in st.session_state:
    st.session_state.explanation_text = None
if 'shap_waterfall_plot' not in st.session_state: # New state for the waterfall plot
    st.session_state.shap_waterfall_plot = None
if 'df_input_original' not in st.session_state:
    st.session_state.df_input_original = None
    
# =============================================================================
# 5. UI: STYLING, TITLES, AND INPUT WIDGETS
# =============================================================================
st.markdown("""<style>...your css...</style>""", unsafe_allow_html=True)
st.markdown('<div class="title-font">Obesity Risk Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="cute-subtitle">A personalized obesity risk assessment with model explanations.</div>', unsafe_allow_html=True)

main_col, explanation_col = st.columns([1, 1])

with main_col:
    st.markdown('<div class="section-title">📝 Input Your Health Data</div>', unsafe_allow_html=True)
    # Full List of Input Widgets
    Age = st.number_input("Age (years)", 14, 100, 25, 1)
    Weight = st.number_input("Weight (kg)", 30.0, 180.0, 70.0, 0.5)
    FCVC = st.slider("Frequency of vegetable consumption (1=low, 3=high)", 1.0, 3.0, 1.0, 0.1)
    NCP = st.slider("Number of main meals per day (1-4)", 1.0, 4.0, 4.0, 0.1)
    CH2O = st.slider("Daily water intake (1=low, 3=high)", 1.0, 3.0, 1.0, 0.1)
    FAF = st.slider("Physical activity frequency per week (0=none, 3=frequent)", 0.0, 3.0, 0.0, 0.1)
    TUE = st.slider("Daily screen time (0=low, 2=high)", 0.0, 2.0, 2.0, 0.1)
    Gender = st.selectbox("Gender", ["Male", "Female"])
    family_history_with_overweight = st.selectbox("Family history of overweight", ["yes", "no"])
    FAVC = st.selectbox("Do you consume high caloric food frequently?", ["yes", "no"], index=0)
    CAEC = st.selectbox("Snacking frequency", ["no", "Sometimes", "Frequently", "Always"], index=3)
    CALC = st.selectbox("Alcohol consumption", ["no", "Sometimes", "Frequently", "Always"], index=3)
    MTRANS = st.selectbox("Primary transportation method", ["Public_Transportation", "Automobile", "Walking", "Motorbike", "Bike"], index=1)


# =============================================================================
# 6. PREDICTION LOGIC
# =============================================================================
# --- Prediction Button and its Logic ---
    if st.button("🔍 Predict Obesity Risk", use_container_width=True):
        input_dict = {"Gender": 1 if Gender == "Male" else 0, "Age": Age, "Weight": Weight, "family_history_with_overweight": 1 if family_history_with_overweight == "yes" else 0, "FAVC": 1 if FAVC == "yes" else 0, "FCVC": FCVC, "NCP": NCP, "CH2O": CH2O, "FAF": FAF, "TUE": TUE, "CAEC_Always": 1 if CAEC == "Always" else 0, "CAEC_Frequently": 1 if CAEC == "Frequently" else 0, "CAEC_Sometimes": 1 if CAEC == "Sometimes" else 0, "CAEC_no": 1 if CAEC == "no" else 0, "CALC_Always": 1 if CALC == "Always" else 0, "CALC_Frequently": 1 if CALC == "Frequently" else 0, "CALC_Sometimes": 1 if CALC == "Sometimes" else 0, "CALC_no": 1 if CALC == "no" else 0, "MTRANS_Automobile": 1 if MTRANS == "Automobile" else 0, "MTRANS_Bike": 1 if MTRANS == "Bike" else 0, "MTRANS_Motorbike": 1 if MTRANS == "Motorbike" else 0, "MTRANS_Public_Transportation": 1 if MTRANS == "Public_Transportation" else 0, "MTRANS_Walking": 1 if MTRANS == "Walking" else 0}
        df_input = pd.DataFrame([input_dict])[feature_names]
        df_input_scaled = df_input.copy()
        df_input_scaled[numeric_cols] = scaler.transform(df_input_scaled[numeric_cols])
        prediction_code = model.predict(df_input_scaled)[0]
        st.session_state.prediction_label = label_mapping.get(prediction_code, "Unknown")
        st.session_state.df_input_original = df_input
        st.session_state.explanation_text = None
        st.session_state.shap_waterfall_plot = None # Reset plot for new prediction

# =============================================================================
# 7. DISPLAY RESULTS AND LOCAL (TEXT) EXPLANATION
# =============================================================================
with explanation_col:
    st.markdown('<div class="section-title">💡 Prediction & Explanation</div>', unsafe_allow_html=True)
    
    # --- LOGIC FIX IS HERE ---
    # First, check if a prediction has been made.
    if st.session_state.prediction_label:
        st.success(f"✅ Your predicted obesity category is: **{st.session_state.prediction_label}**")

        if st.button("📊 Explain My Prediction", use_container_width=True):
            narrative_rules = [
                {'feature': 'FCVC', 'condition': lambda v: v < 1.5, 'type': 'risk', 'text': 'Your **vegetable consumption is low**, which can impact metabolic health.'},
                {'feature': 'FAF', 'condition': lambda v: v < 1.0, 'type': 'risk', 'text': 'Your **frequency of physical activity is low**, a key factor in weight management.'},
                {'feature': 'TUE', 'condition': lambda v: v > 1.5, 'type': 'risk', 'text': 'Your **daily screen time is long**, often linked to a sedentary lifestyle.'},
                {'feature': 'FAVC', 'condition': lambda v: v == 1, 'type': 'risk', 'text': 'You **frequently consume high-caloric food**, directly impacting energy balance.'},
                {'feature': 'CH2O', 'condition': lambda v: v < 1.5, 'type': 'risk', 'text': 'Your **daily water intake may be insufficient**.'},
                {'feature': 'NCP', 'condition': lambda v: v > 3.5, 'type': 'risk', 'text': 'Your **number of main meals per day is high**.'},
                {'feature': 'family_history_with_overweight', 'condition': lambda v: v == 1, 'type': 'risk', 'text': 'You have a **family history of overweight**.'},
                {'feature': 'CAEC_Always', 'condition': lambda v: v == 1, 'type': 'risk', 'text': 'You **always snack between meals**.'},
                {'feature': 'CAEC_Frequently', 'condition': lambda v: v == 1, 'type': 'risk', 'text': 'You **frequently snack between meals**.'},
                {'feature': 'CALC_Always', 'condition': lambda v: v == 1, 'type': 'risk', 'text': 'You **always consume alcohol**.'},
                {'feature': 'CALC_Frequently', 'condition': lambda v: v == 1, 'type': 'risk', 'text': 'You **frequently consume alcohol**.'},
                {'feature': 'MTRANS_Automobile', 'condition': lambda v: v == 1, 'type': 'risk', 'text': 'Your primary transport is by **automobile** (sedentary).'},
                {'feature': 'FAF', 'condition': lambda v: v > 2.5, 'type': 'protective', 'text': 'You maintain a **high frequency of physical activity**.'},
                {'feature': 'FCVC', 'condition': lambda v: v > 2.5, 'type': 'protective', 'text': 'You **frequently consume vegetables**.'},
                {'feature': 'TUE', 'condition': lambda v: v < 0.5, 'type': 'protective', 'text': 'Your **daily screen time is very short**.'},
                {'feature': 'MTRANS_Walking', 'condition': lambda v: v == 1, 'type': 'protective', 'text': 'Your primary transport is **walking**.'},
            ]

            with st.spinner('Analyzing your health profile and generating insights...'):
                try:
                    df_scaled = pd.DataFrame(st.session_state.df_input_original, columns=feature_names)
                    df_scaled[numeric_cols] = scaler.transform(df_scaled[numeric_cols])
                    
                    explainer = shap.Explainer(model)
                    explanation = explainer(df_scaled) # Use the modern Explanation object
                    
                    predicted_class_index = st.session_state.df_input_original.index[0] # Should be 0 for a single prediction
                    
                    # --- NEW: Generate the Waterfall plot ---
                    fig, ax = plt.subplots()
                    # We select the explanation for the specific predicted class
                    shap.plots.waterfall(explanation[predicted_class_index, :, model.predict(df_scaled)[0]], show=False)
                    plt.title(f"How Your Inputs Led to the '{st.session_state.prediction_label}' Prediction")
                    plt.tight_layout()
                    st.session_state.shap_waterfall_plot = fig
                    # --- END OF PLOT GENERATION ---
                    
                    # --- NARRATIVE GENERATION (This logic can be reused) ---
                    shap_df = pd.DataFrame({
                        'feature': feature_names,
                        'shap_value': explanation.values[0, :, predicted_class_index],
                        'feature_value': st.session_state.df_input_original.iloc[0].values
                    }).set_index('feature')
                    shap_df['abs_shap_value'] = shap_df['shap_value'].abs()
                    ranked_features = shap_df.sort_values('abs_shap_value', ascending=False)
                    
                    # 2. Generate the narrative text
                    pred_label = st.session_state.prediction_label
                    risk_narratives = []
                    protective_narratives = []

                    age_val = st.session_state.df_input_original['Age'].iloc[0]
                    weight_val = st.session_state.df_input_original['Weight'].iloc[0]
                    opening_statement = f"Based on the information you provided, the model predicted your risk category as **{pred_label}**.\n\n"
                    if 'Weight' in ranked_features.index[:2] and 'Age' in ranked_features.index[:2]:
                         opening_statement += f"For an individual of **{age_val:.0f} years old**, a weight of **{weight_val:.0f} kg** was the most significant factor leading to this prediction."
                    else:
                        primary_driver_name = ranked_features.index[0]
                        primary_driver_text = primary_driver_name.replace("_", " ").title()
                        opening_statement += f"Among your various inputs, **{primary_driver_text}** was the most impactful factor for this prediction."

                    shap_driven_narratives = []
                    mentioned_features = set()
                    for feature, row in ranked_features.head(4).iterrows():
                        is_risk = row['shap_value'] > 0
                        for rule in narrative_rules:
                            if rule['feature'] == feature:
                                if (is_risk and rule['type'] == 'risk' and rule['condition'](row['feature_value'])) or \
                                   (not is_risk and rule['type'] == 'protective' and rule['condition'](row['feature_value'])):
                                    shap_driven_narratives.append(rule['text'])
                                    mentioned_features.add(feature)
                                    break
                    
                    other_narratives = []
                    for rule in narrative_rules:
                        feature = rule['feature']
                        if feature not in mentioned_features and rule['condition'](st.session_state.df_input_original.iloc[0][feature]):
                            other_narratives.append(rule['text'])
                    
                    final_explanation = opening_statement
                    if shap_driven_narratives:
                        final_explanation += "\n\n**Model's Key Insights (in order of impact):**\n"
                        for text in shap_driven_narratives:
                            final_explanation += f"- {text}\n"
                    if other_narratives:
                        final_explanation += "\n**Other Noteworthy Health Habits:**\n"
                        for text in other_narratives:
                            final_explanation += f"- {text}\n"
                    
                    st.session_state.explanation_text = final_explanation
                    # --- END OF NARRATIVE GENERATION ---

                except Exception as e:
                    st.error(f"Sorry, the explanation could not be generated: {e}")
        
        # --- Display logic ---
        if st.session_state.shap_waterfall_plot:
            st.pyplot(st.session_state.shap_waterfall_plot)
        
        if st.session_state.explanation_text:
            st.info(st.session_state.explanation_text)
            
    else:
        st.info("Please input your data and click 'Predict' to see the results.")
# =============================================================================
# 9. FOOTER
# =============================================================================
st.markdown('<div class="footer">Made with ❤️ by Your AI Assistant</div>', unsafe_allow_html=True)
