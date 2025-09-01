
# =============================================================================
# FINAL APP: try04.py (Single-Page Version)
# This version uses a state-driven approach to switch between views
# within a single script, which is more robust than st.switch_page.
# =============================================================================

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
# 2. LOAD ARTIFACTS AND CONFIGURE API (Cached for performance)
# =============================================================================
@st.cache_resource
def load_artifacts():
    """
    Loads all necessary files once and caches them for the session.
    """
    try:
        model = joblib.load("xgb_obesity_model.pkl")
        scaler = joblib.load("scaler.pkl")
        feature_names = model.feature_names_in_
        
        if "GEMINI_API_KEY" in st.secrets:
            genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        else:
            st.error("Gemini API key not found. Please add it to your Streamlit secrets.")
            st.stop()
            
        return model, scaler, feature_names
    except Exception as e:
        st.error(f"Error loading necessary files: {e}")
        st.stop()

model, scaler, feature_names = load_artifacts()

# =============================================================================
# 3. GLOBAL VARIABLES / CONSTANTS & HELPER FUNCTIONS
# =============================================================================
numeric_cols = ["Age", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]
label_mapping = {
    0: "Insufficient_Weight", 1: "Normal_Weight", 2: "Overweight_Level_I",
    3: "Overweight_Level_II", 4: "Obesity_Type_I", 5: "Obesity_Type_II",
    6: "Obesity_Type_III"
}
one_hot_prefixes = ['CAEC_', 'CALC_', 'MTRANS_']

# --- Helper function to generate the personalized "Risk Factors" bar plot ---
@st.cache_data
def generate_risk_factors_plot(_shap_df, prediction_label):
    risk_features = _shap_df[_shap_df['shap_value'] > 0.05].copy()
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
    fig, ax = plt.subplots(figsize=(10, len(risk_features_to_plot) * 0.4 + 1.5))
    ax.barh(risk_features_to_plot.index, risk_features_to_plot['shap_value'], color='#ff4d4d')
    ax.set_xlabel("Positive Impact on Prediction (SHAP value)")
    ax.set_title(f"Main Risk Factors for '{prediction_label}' Prediction")
    plt.tight_layout()
    return fig

# --- Helper function to call Gemini API for a narrative explanation ---
@st.cache_data
def generate_narrative_with_gemini(raw_explanation_data):
    llm = genai.GenerativeModel('gemini-1.5-flash')
    prompt = f"""
    As a professional and empathetic health consultant, rewrite the following mechanical analysis into a warm, encouraging, and easy-to-understand health report.
    **Instructions:**
    1. Start with a clear summary of the prediction.
    2. Analyze the **"Model's Key Insights"**. Explain them contextually.
    3. Discuss the **"Other Noteworthy Health Habits"**.
    4. Conclude with a **"Recommendations"** section with 2-3 actionable suggestions.
    5. Maintain a positive, non-judgmental tone. Use Markdown for formatting.
    **Mechanical Analysis to Transform:**
    ---
    {raw_explanation_data}
    ---
    """
    try:
        response = llm.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Sorry, the AI-powered explanation could not be generated. Error: {e}"

# =============================================================================
# 4. PAGE DEFINITIONS (as functions)
# =============================================================================
def render_input_page():
    """
    Renders all the UI and logic for the data input page.
    """
    
    #st.markdown('<div class="title-font">Obesity Risk Predictor</div>', unsafe_allow_html=True)
    #st.markdown('<div class="cute-subtitle">A personalized obesity risk assessment with AI-powered explanations.</div>', unsafe_allow_html=True)

    st.markdown("""<style>
    .title-font { font-size: 48px; font-weight: bold; color: #2c3e50; text-align: center; font-family: 'Arial Rounded MT Bold', sans-serif; }
    .cute-subtitle { font-size: 22px; font-family: "Comic Sans MS", cursive, sans-serif; color: #e17055; text-align: center; margin-bottom: 30px; }
    .section-title { font-size: 24px; font-weight: bold; color: #0984e3; margin-top: 40px; margin-bottom: 20px; }
    .footer { margin-top: 60px; text-align: center; font-size: 14px; color: #636e72; }
    </style>""", unsafe_allow_html=True)
    st.markdown('<div class="title-font">Obesity Risk Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="cute-subtitle">A personalized obesity risk assessment with AI-powered explanations.</div>', unsafe_allow_html=True)



    with st.form("user_input_form"):
        st.markdown('<div class="section-title">📝 Input Your Health Data</div>', unsafe_allow_html=True)
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

        submitted = st.form_submit_button("🔍 Analyze My Risk Profile")

        if submitted:
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
            
            # Store the user input and set the view to 'report'
            st.session_state.user_input = input_dict
            st.session_state.view = 'report'
            st.rerun() # Rerun the script to show the report page

def render_report_page():
    """
    Renders the full report page with plot and text explanations.
    """

    st.markdown("""<style>
    .section-title { font-size: 24px; font-weight: bold; color: #0984e3; margin-top: 40px; margin-bottom: 20px; }
    .footer { margin-top: 60px; text-align: center; font-size: 14px; color: #636e72; }
    </style>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Your Personalized Health Report</div>', unsafe_allow_html=True)
    
    # Retrieve user input from session state
    user_input = st.session_state.user_input
    
    with st.spinner("Analyzing your profile and generating your report..."):
        df_input = pd.DataFrame([user_input])[feature_names]
        df_input_scaled = df_input.copy()
        df_input_scaled[numeric_cols] = scaler.transform(df_input_scaled[numeric_cols])
        prediction_code = model.predict(df_input_scaled)[0]
        prediction_label = label_mapping.get(prediction_code, "Unknown")

        explainer = shap.Explainer(model)
        shap_values = explainer.shap_values(df_input_scaled)
        shap_df = pd.DataFrame({'feature': feature_names, 'shap_value': shap_values[0, :, prediction_code], 'feature_value': df_input.iloc[0].values}).set_index('feature')

        # Generate the personalized plot
        risk_plot = generate_risk_factors_plot(shap_df, prediction_label)

        # --- Generate the "Raw Intelligence" for Gemini ---
        shap_df['abs_shap_value'] = shap_df['shap_value'].abs()
        ranked_features = shap_df.sort_values('abs_shap_value', ascending=False)
        age_val, weight_val, gender_code = df_input['Age'].iloc[0], df_input['Weight'].iloc[0], df_input['Gender'].iloc[0]
        gender_text = "female" if gender_code == 0 else "male"
        raw_text_for_ai = f"PREDICTION RESULT: {prediction_label}\nUSER PROFILE: {age_val:.0f} years old {gender_text}, {weight_val:.0f} kg\n\n"
        
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
        
        shap_driven_narratives, mentioned_features = [], set()
        for feature, row in ranked_features.head(4).iterrows():
            is_risk = row['shap_value'] > 0
            for rule in narrative_rules:
                if rule['feature'] == feature:
                    if (is_risk and rule['type'] == 'risk' and rule['condition'](row['feature_value'])) or (not is_risk and rule['type'] == 'protective' and rule['condition'](row['feature_value'])):
                        shap_driven_narratives.append(rule['text'])
                        mentioned_features.add(feature)
                        break
        other_narratives = []
        for rule in narrative_rules:
            feature = rule['feature']
            if feature not in mentioned_features and rule['condition'](df_input.iloc[0][feature]):
                other_narratives.append(rule['text'])
                
        if shap_driven_narratives:
            raw_text_for_ai += "Model's Key Insights (in order of impact):\n- " + "\n- ".join(shap_driven_narratives)
        if other_narratives:
            raw_text_for_ai += "\n\nOther Noteworthy Health Habits:\n- " + "\n- ".join(other_narratives)
        
        # Get the AI-powered narrative
        narrative_text = generate_narrative_with_gemini(raw_text_for_ai)

    # --- Display the Final Report in the "Top-Down" layout ---
    st.success(f"✅ Your predicted obesity category is: **{prediction_label}**")
    
    st.markdown("#### Main Influential Factors (Personalized Chart)")
    if risk_plot:
        st.pyplot(risk_plot)
    else:
        st.write("No significant risk factors were identified by the model for this prediction.")

    st.markdown("#### AI-Powered Health Analysis")
    st.info(narrative_text)
    
    # Add a button to go back to the input page
    if st.button("⬅️ Start Over"):
        st.session_state.view = 'input'
        st.rerun()

# =============================================================================
# SCRIPT EXECUTION ROUTER
# =============================================================================
# This is the main router of the app. It checks the 'view' state and calls
# the appropriate function to render the page.
if 'view' not in st.session_state:
    st.session_state.view = 'input'

if st.session_state.view == 'input':
    render_input_page()
elif st.session_state.view == 'report':
    render_report_page()
else:
    st.session_state.view = 'input' # Default to input page if state is invalid
    st.rerun()

st.markdown('<div class="footer">Made with ❤️ by Your AI Assistant</div>', unsafe_allow_html=True)
