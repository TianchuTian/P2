# =============================================================================
# 1. IMPORT LIBRARIES
# =============================================================================
import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import google.generativeai as genai

# =============================================================================
# 2. LOAD ARTIFACTS AND CONFIGURE API
# =============================================================================
# This block loads all necessary files. The app will stop if any are missing.
try:
    model = joblib.load("xgb_obesity_model.pkl")
    scaler = joblib.load("scaler.pkl")
    # Get feature names directly from the trained model object for robustness.
    feature_names = model.feature_names_in_
    
    # --- Configure the Gemini API ---
    # The API key should be stored securely in Streamlit's secrets management.
    # On your Streamlit Community Cloud, go to Settings -> Secrets and add a secret:
    GEMINI_API_KEY = "AIzaSyB27YtzoKdmgkTxmw_U90_uN08NsSdS2Rk"
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

except FileNotFoundError:
    st.error("Error: Model or scaler files not found. Please ensure all necessary files are in your GitHub repository.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading files or configuring the API: {e}")
    st.stop()

# =============================================================================
# 3. GLOBAL VARIABLES / CONSTANTS
# =============================================================================
# Define the list of numeric columns that require scaling.
numeric_cols = ["Age", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]

# Mapping from the model's numeric output to human-readable labels.
label_mapping = {
    0: "Insufficient_Weight", 1: "Normal_Weight", 2: "Overweight_Level_I",
    3: "Overweight_Level_II", 4: "Obesity_Type_I", 5: "Obesity_Type_II",
    6: "Obesity_Type_III"
}
# Define prefixes for one-hot encoded features to handle them smartly in the plot.
one_hot_prefixes = ['CAEC_', 'CALC_', 'MTRANS_']

# =============================================================================
# 4. PAGE CONFIGURATION AND SESSION STATE INITIALIZATION
# =============================================================================
st.set_page_config(page_title="Obesity Risk Predictor", page_icon="🍔", layout="wide")

# Initialize session state variables to remember values across reruns.
if 'prediction_label' not in st.session_state:
    st.session_state.prediction_label = None
if 'explanation_text' not in st.session_state:
    st.session_state.explanation_text = None
if 'risk_factors_plot' not in st.session_state:
    st.session_state.risk_factors_plot = None
if 'df_input_original' not in st.session_state:
    st.session_state.df_input_original = None

# =============================================================================
# 5. UI: STYLING, TITLES, AND INPUT WIDGETS
# =============================================================================
st.markdown("""<style>
.title-font { font-size: 48px; font-weight: bold; color: #2c3e50; text-align: center; font-family: 'Arial Rounded MT Bold', sans-serif; }
.cute-subtitle { font-size: 22px; font-family: "Comic Sans MS", cursive, sans-serif; color: #e17055; text-align: center; margin-bottom: 30px; }
.section-title { font-size: 24px; font-weight: bold; color: #0984e3; margin-top: 40px; margin-bottom: 20px; }
.footer { margin-top: 60px; text-align: center; font-size: 14px; color: #636e72; }
</style>""", unsafe_allow_html=True)
st.markdown('<div class="title-font">Obesity Risk Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="cute-subtitle">A personalized obesity risk assessment with AI-powered explanations.</div>', unsafe_allow_html=True)

# Use columns for a cleaner layout: left for inputs, right for outputs.
main_col, explanation_col = st.columns([1, 1])

with main_col:
    st.markdown('<div class="section-title">📝 Input Your Health Data</div>', unsafe_allow_html=True)
    
    # --- Full List of Input Widgets ---
    Age = st.number_input("Age (years)", 14, 100, 25, 1)
    Weight = st.number_input("Weight (kg)", 30.0, 180.0, 70.0, 0.5)
    FCVC = st.slider("[FCVC] Frequency of vegetable consumption (1=low, 3=high)", 1.0, 3.0, 1.0, 0.1)
    NCP = st.slider("[NCP] Number of main meals per day (1-4)", 1.0, 4.0, 4.0, 0.1)
    CH2O = st.slider("[CH2O] Daily water intake (1=low, 3=high)", 1.0, 3.0, 1.0, 0.1)
    FAF = st.slider("[FAF] Physical activity frequency per week (0=none, 3=frequent)", 0.0, 3.0, 0.0, 0.1)
    TUE = st.slider("[TUE] Daily screen time (0=low, 2=high)", 0.0, 2.0, 2.0, 0.1)
    Gender = st.selectbox("Gender", ["Male", "Female"])
    family_history_with_overweight = st.selectbox("Family history of overweight", ["yes", "no"])
    FAVC = st.selectbox("[FAVC] Do you consume high caloric food frequently?", ["yes", "no"], index=0)
    CAEC = st.selectbox("[CAEC] Snacking frequency", ["no", "Sometimes", "Frequently", "Always"], index=3)
    CALC = st.selectbox("[CALC] Alcohol consumption", ["no", "Sometimes", "Frequently", "Always"], index=3)
    MTRANS = st.selectbox("[MTRANS] Primary transportation method", ["Public_Transportation", "Automobile", "Walking", "Motorbike", "Bike"], index=1)

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
        st.session_state.prediction_label = label_mapping.get(prediction_code, "Unknown")
        st.session_state.df_input_original = df_input
        st.session_state.explanation_text = None
        st.session_state.risk_factors_plot = None

# =============================================================================
# 6. EXPLANATION GENERATION AND DISPLAY LOGIC
# =============================================================================

# --- Function to generate the personalized "Risk Factors" bar plot ---
def generate_risk_factors_plot(shap_df, prediction_label):
    risk_features = shap_df[shap_df['shap_value'] > 0.05].copy()
    intuitive_risk_features = []
    for feature, row in risk_features.iterrows():
        is_one_hot = any(prefix in feature for prefix in one_hot_prefixes)
        if is_one_hot:
            if row['feature_value'] == 1:
                intuitive_risk_features.append(feature)
        else:
            intuitive_risk_features.append(feature)
    risk_features_to_plot = risk_features.loc[intuitive_risk_features]
    if risk_features_to_plot.empty:
        return None
    risk_features_to_plot = risk_features_to_plot.sort_values('shap_value', ascending=True)
    fig, ax = plt.subplots(figsize=(8, len(risk_features_to_plot) * 0.5 + 1))
    ax.barh(risk_features_to_plot.index, risk_features_to_plot['shap_value'], color='#ff4d4d')
    ax.set_xlabel("Positive Impact on Prediction (SHAP value)")
    ax.set_title(f"Main Risk Factors for '{prediction_label}' Prediction")
    plt.tight_layout()
    return fig

# --- Function to call Gemini API for a narrative explanation ---
def generate_narrative_with_gemini(raw_explanation_data):
    """
    Takes the raw, rule-based explanation and sends it to the Gemini API
    to get a more human-like, insightful narrative.
    """
    llm = genai.GenerativeModel('gemini-1.5-flash')
    prompt = f"""
    As a professional and empathetic health consultant, your task is to rewrite the following mechanical analysis into a warm, encouraging, and easy-to-understand health report for a user.

    **Instructions:**
    1.  Start with a clear summary of the prediction.
    2.  First, analyze the **"Model's Key Insights"**. These are the most impactful factors the model used. Explain them in a contextual way (e.g., "For your age, your weight is...").
    3.  Next, discuss the **"Other Noteworthy Health Habits"**. These are other risk factors the user entered that they should be aware of, even if they weren't the top drivers for this specific prediction.
    4.  Conclude with a **"Recommendations"** section, providing 2-3 simple, actionable suggestions based on ALL the risk factors identified.
    5.  Maintain a positive, non-judgmental, and motivating tone throughout. Use Markdown for formatting (bolding, bullet points).

    **Here is the mechanical analysis to transform:**
    ---
    {raw_explanation_data}
    ---
    """
    try:
        response = llm.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Sorry, the AI-powered explanation could not be generated at this time. Error: {e}"

with explanation_col:
    st.markdown('<div class="section-title">💡 Prediction & Explanation</div>', unsafe_allow_html=True)
    if st.session_state.prediction_label:
        st.success(f"✅ Your predicted obesity category is: **{st.session_state.prediction_label}**")

        if st.button("📊 Explain My Prediction", use_container_width=True):
            # The comprehensive narrative rules engine
            narrative_rules = [
                {'feature': 'FCVC', 'condition': lambda v: v < 1.5, 'type': 'risk', 'text': 'Your **vegetable consumption is low**, and a lack of dietary diversity can increase the risk of obesity and metabolic syndrome.'},
                {'feature': 'FAF', 'condition': lambda v: v < 1.0, 'type': 'risk', 'text': 'Your **frequency of physical activity is low**. Increasing regular exercise helps boost metabolism and control weight.'},
                {'feature': 'TUE', 'condition': lambda v: v > 1.5, 'type': 'risk', 'text': 'Your **daily screen time is long**, which is often associated with sedentary behavior and is a risk factor for weight gain.'},
                {'feature': 'FAVC', 'condition': lambda v: v == 1, 'type': 'risk', 'text': 'You **frequently consume high-caloric food**, which is a direct cause of excessive energy intake and weight gain.'},
                {'feature': 'CH2O', 'condition': lambda v: v < 1.5, 'type': 'risk', 'text': 'Your **daily water intake may be insufficient**. Adequate hydration helps promote metabolism.'},
                {'feature': 'NCP', 'condition': lambda v: v > 3.5, 'type': 'risk', 'text': 'Your **number of main meals per day is high**, which may lead to a higher total daily calorie intake.'},
                {'feature': 'family_history_with_overweight', 'condition': lambda v: v == 1, 'type': 'risk', 'text': 'You have a **family history of overweight**, which means you may need to be more mindful of your lifestyle to maintain a healthy weight.'},
                {'feature': 'CAEC_Always', 'condition': lambda v: v == 1, 'type': 'risk', 'text': 'You **always snack between meals**, which significantly increases additional calorie intake.'},
                {'feature': 'CAEC_Frequently', 'condition': lambda v: v == 1, 'type': 'risk', 'text': 'You **frequently snack between meals**, which adds extra calories to your diet.'},
                {'feature': 'CALC_Always', 'condition': lambda v: v == 1, 'type': 'risk', 'text': 'You **always consume alcohol**, which is high in calories and significantly increases obesity risk.'},
                {'feature': 'CALC_Frequently', 'condition': lambda v: v == 1, 'type': 'risk', 'text': 'You **frequently consume alcohol**, which is high in calories and increases obesity risk.'},
                {'feature': 'MTRANS_Automobile', 'condition': lambda v: v == 1, 'type': 'risk', 'text': 'Your primary transport is by **automobile**, which is a sedentary behavior that reduces daily physical activity.'},
                
                # Protective factors
                {'feature': 'FAF', 'condition': lambda v: v > 2.5, 'type': 'protective', 'text': 'You maintain a **high frequency of physical activity**, which is a crucial protective factor for maintaining a healthy weight.'},
                {'feature': 'FCVC', 'condition': lambda v: v > 2.5, 'type': 'protective', 'text': 'You **frequently consume vegetables**, which is an excellent dietary habit that helps control calories and provide essential nutrients.'},
                {'feature': 'TUE', 'condition': lambda v: v < 0.5, 'type': 'protective', 'text': 'Your **daily screen time is very short**, which often implies a more active lifestyle.'},
                {'feature': 'MTRANS_Walking', 'condition': lambda v: v == 1, 'type': 'protective', 'text': 'Your primary transport is **walking**, which is an excellent habit that effectively increases daily energy expenditure.'},
            ]


            with st.spinner('Analyzing your health profile and generating AI-powered insights...'):
                try:
                    df_scaled = pd.DataFrame(st.session_state.df_input_original, columns=feature_names)
                    df_scaled[numeric_cols] = scaler.transform(df_scaled[numeric_cols])
                    
                    explainer = shap.Explainer(model)
                    shap_values = explainer.shap_values(df_scaled)
                    
                    predicted_class_index = model.predict(df_scaled)[0]
                    
                    shap_df = pd.DataFrame({
                        'feature': feature_names,
                        'shap_value': shap_values[0, :, predicted_class_index],
                        'feature_value': st.session_state.df_input_original.iloc[0].values
                    }).set_index('feature')
                    
                    st.session_state.risk_factors_plot = generate_risk_factors_plot(shap_df, st.session_state.prediction_label)
                    
                    # --- GENERATE THE "RAW INTELLIGENCE" FOR GEMINI ---
                    # 1.
