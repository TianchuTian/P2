# =============================================================================
# FILE 2: pages/Report.py
# This is the report page for displaying results and explanations.
# =============================================================================
import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import google.generativeai as genai

# Set page configuration FIRST.
st.set_page_config(page_title="Your Health Report", page_icon="📊", layout="wide")

# =============================================================================
# LOAD ARTIFACTS AND HELPER FUNCTIONS (self-contained for this page)
# =============================================================================
@st.cache_resource
def load_artifacts():
    """
    Loads all necessary files once and caches them for the session.
    This is a performance optimization.
    """
    try:
        model = joblib.load("xgb_obesity_model.pkl")
        scaler = joblib.load("scaler.pkl")
        feature_names = model.feature_names_in_
        # Configure the Gemini API if the key is available
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
    
# Global variables and helper functions needed for this page
numeric_cols = ["Age", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]
label_mapping = {
    0: "Insufficient_Weight", 1: "Normal_Weight", 2: "Overweight_Level_I",
    3: "Overweight_Level_II", 4: "Obesity_Type_I", 5: "Obesity_Type_II",
    6: "Obesity_Type_III"
}
one_hot_prefixes = ['CAEC_', 'CALC_', 'MTRANS_']

# --- Function to generate the personalized "Risk Factors" bar plot ---
@st.cache_data
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
    fig, ax = plt.subplots(figsize=(10, len(risk_features_to_plot) * 0.5 + 1.5))
    ax.barh(risk_features_to_plot.index, risk_features_to_plot['shap_value'], color='#ff4d4d')
    ax.set_xlabel("Positive Impact on Prediction (SHAP value)")
    ax.set_title(f"Main Risk Factors for '{prediction_label}' Prediction")
    plt.tight_layout()
    return fig

# --- Function to call Gemini API for a narrative explanation ---
@st.cache_data
def generate_narrative_with_gemini(raw_explanation_data):
    llm = genai.GenerativeModel('gemini-1.5-flash')
    prompt = f"""
    As a professional and empathetic health consultant, rewrite the following mechanical analysis into a warm, encouraging, and easy-to-understand health report.

    **Instructions:**
    1. Start with a clear summary of the prediction.
    2. Analyze the **"Model's Key Insights"**. Explain them contextually (e.g., "For your age, your weight is...").
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
# MAIN PAGE LOGIC
# =============================================================================
st.markdown('<div class="title-font">Your Personalized Health Report</div>', unsafe_allow_html=True)

# Check if user input exists in session state. If not, guide the user back to the home page.
if 'user_input' not in st.session_state or not st.session_state.user_input:
    st.warning("Please go to the Home page to input your data first.")
    st.page_link("app.py", label="Go to Home Page", icon="🏠")
    st.stop()

# --- Retrieve user input and perform analysis (this runs once when the page loads) ---
with st.spinner("Analyzing your profile and generating your report..."):
    # Create DataFrame from the stored user input
    df_input = pd.DataFrame([st.session_state.user_input])[feature_names]
    
    # Scale data and make prediction
    df_input_scaled = df_input.copy()
    df_input_scaled[numeric_cols] = scaler.transform(df_input_scaled[numeric_cols])
    prediction_code = model.predict(df_input_scaled)[0]
    prediction_label = label_mapping.get(prediction_code, "Unknown")

    # Generate SHAP analysis
    explainer = shap.Explainer(model)
    shap_values = explainer.shap_values(df_input_scaled)
    shap_df = pd.DataFrame({
        'feature': feature_names,
        'shap_value': shap_values[0, :, prediction_code],
        'feature_value': df_input.iloc[0].values
    }).set_index('feature')

    # Generate the personalized plot
    risk_plot = generate_risk_factors_plot(shap_df, prediction_label)

    # --- Generate the "Raw Intelligence" for Gemini ---
    # (This is the full, correct logic from our previous final version)
    shap_df['abs_shap_value'] = shap_df['shap_value'].abs()
    ranked_features = shap_df.sort_values('abs_shap_value', ascending=False)
    
    age_val = df_input['Age'].iloc[0]
    weight_val = df_input['Weight'].iloc[0]
    gender_code = df_input['Gender'].iloc[0]
    gender_text = "female" if gender_code == 0 else "male"
    
    raw_text_for_ai = f"PREDICTION RESULT: {prediction_label}\n"
    raw_text_for_ai += f"USER PROFILE: {age_val:.0f} years old {gender_text}, {weight_val:.0f} kg\n\n"
    
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
        if feature not in mentioned_features and rule['condition'](df_input.iloc[0][feature]):
            other_narratives.append(rule['text'])
            
    if shap_driven_narratives:
        raw_text_for_ai += "Model's Key Insights (in order of impact):\n- " + "\n- ".join(shap_driven_narratives)
    if other_narratives:
        raw_text_for_ai += "\n\nOther Noteworthy Health Habits:\n- " + "\n- ".join(other_narratives)
    
    # Get the AI-powered narrative
    narrative_text = generate_narrative_with_gemini(raw_text_for_ai)

# --- Display the Final Report in the desired "Top-Down" layout ---
st.success(f"✅ Your predicted obesity category is: **{prediction_label}**")

st.markdown("#### Main Influential Factors (Personalized Chart)")
if risk_plot:
    st.pyplot(risk_plot)
else:
    st.write("No significant risk factors were identified by the model for this prediction.")

st.markdown("#### AI-Powered Health Analysis")
st.info(narrative_text)

st.markdown('<div class="footer">Made with ❤️ by Your AI Assistant</div>', unsafe_allow_html=True)
