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

# --- NEW: Function to generate the personalized Health Dashboard plot ---
def generate_health_card_html(df_input, rules):
    """
    Generates a clean, readable HTML table to act as a health scorecard.
    """
    card_html = """
    <style>
        .card { border-radius: 10px; padding: 15px; background-color: #f8f9fa; }
        .card-title { font-weight: bold; font-size: 20px; color: #333; margin-bottom: 15px; }
        .habit-row { display: flex; align-items: center; padding: 8px 0; border-bottom: 1px solid #e9ecef; }
        .habit-icon { font-size: 24px; margin-right: 15px; }
        .habit-text { flex-grow: 1; color: #555; }
        .habit-value { font-weight: bold; color: #000; }
        .risk { color: #e74c3c; }
        .protective { color: #2ecc71; }
    </style>
    <div class="card">
        <div class="card-title">Your Personalized Health Habits Scorecard</div>
    """
    
    evaluated_items = []
    
    # Iterate through rules to evaluate each user input
    for rule in rules:
        feature = rule['feature']
        value = df_input.iloc[0][feature]
        
        if rule['condition'](value):
            is_risk = rule['type'] == 'risk'
            icon = '⚠️' if is_risk else '✅'
            css_class = 'risk' if is_risk else 'protective'
            
            # Use the 'value_text' from the rule to show the user's actual input
            value_display = rule.get('value_text', lambda v: f'{v}')(value)
            
            evaluated_items.append(f"""
            <div class="habit-row">
                <div class="habit-icon">{icon}</div>
                <div class="habit-text">{rule['text']}</div>
                <div class="habit-value {css_class}">{value_display}</div>
            </div>
            """)
            
    if not evaluated_items:
        card_html += "<p>All habits are within the normal range.</p>"
    else:
        card_html += "".join(evaluated_items)
        
    card_html += "</div>"
    return card_html

# =============================================================================
# 4. PAGE CONFIGURATION AND SESSION STATE INITIALIZATION
# =============================================================================
st.set_page_config(page_title="Obesity Risk Predictor", page_icon="🍔", layout="wide") # Use wide layout

if 'prediction_label' not in st.session_state:
    st.session_state.prediction_label = None
if 'explanation_html' not in st.session_state:
    st.session_state.explanation_html = None
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
        st.session_state.explanation_html = None

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
                {'feature': 'FCVC', 'condition': lambda v: v < 1.5, 'type': 'risk', 'text': 'Low vegetable consumption', 'value_text': lambda v: f'Level {v:.1f}'},
                {'feature': 'FCVC', 'condition': lambda v: v > 2.5, 'type': 'protective', 'text': 'High vegetable consumption', 'value_text': lambda v: f'Level {v:.1f}'},
                {'feature': 'FAF', 'condition': lambda v: v < 1.0, 'type': 'risk', 'text': 'Low physical activity', 'value_text': lambda v: f'Level {v:.1f}'},
                {'feature': 'FAF', 'condition': lambda v: v > 2.5, 'type': 'protective', 'text': 'High physical activity', 'value_text': lambda v: f'Level {v:.1f}'},
                {'feature': 'TUE', 'condition': lambda v: v > 1.5, 'type': 'risk', 'text': 'Long daily screen time', 'value_text': lambda v: f'Level {v:.1f}'},
                {'feature': 'FAVC', 'condition': lambda v: v == 1, 'type': 'risk', 'text': 'Frequent high-caloric food', 'value_text': lambda v: 'Yes'},
                {'feature': 'CH2O', 'condition': lambda v: v < 1.5, 'type': 'risk', 'text': 'Low daily water intake', 'value_text': lambda v: f'Level {v:.1f}'},
                {'feature': 'NCP', 'condition': lambda v: v > 3.5, 'type': 'risk', 'text': 'High number of main meals', 'value_text': lambda v: f'Level {v:.1f}'},
                {'feature': 'family_history_with_overweight', 'condition': lambda v: v == 1, 'type': 'risk', 'text': 'Family history of overweight', 'value_text': lambda v: 'Yes'},
                {'feature': 'CAEC_Always', 'condition': lambda v: v == 1, 'type': 'risk', 'text': 'Always snacks', 'value_text': lambda v: 'Yes'},
                {'feature': 'CAEC_Frequently', 'condition': lambda v: v == 1, 'type': 'risk', 'text': 'Frequently snacks', 'value_text': lambda v: 'Yes'},
                {'feature': 'CALC_Always', 'condition': lambda v: v == 1, 'type': 'risk', 'text': 'Always consumes alcohol', 'value_text': lambda v: 'Yes'},
                {'feature': 'CALC_Frequently', 'condition': lambda v: v == 1, 'type': 'risk', 'text': 'Frequently consumes alcohol', 'value_text': lambda v: 'Yes'},
                {'feature': 'MTRANS_Automobile', 'condition': lambda v: v == 1, 'type': 'risk', 'text': 'Transport by automobile', 'value_text': lambda v: 'Yes'},
                {'feature': 'MTRANS_Walking', 'condition': lambda v: v == 1, 'type': 'protective', 'text': 'Transport by walking', 'value_text': lambda v: 'Yes'},
            ]

            with st.spinner('Analyzing your health profile...'):
                st.session_state.explanation_html = generate_health_card_html(st.session_state.df_input_original, narrative_rules)

        # This block now correctly displays the HTML card if it exists.
        # It is still inside the 'if st.session_state.prediction_label' block.
        if st.session_state.explanation_html:
            st.markdown(st.session_state.explanation_html, unsafe_allow_html=True)
            
    # This 'else' statement now correctly corresponds to the 'if st.session_state.prediction_label'
    # It will only show when the app starts and no prediction has been made yet.
    else:
        st.info("Please input your data and click 'Predict' to see the results.")
# =============================================================================
# 9. FOOTER
# =============================================================================
st.markdown('<div class="footer">Made with ❤️ by Your AI Assistant</div>', unsafe_allow_html=True)
