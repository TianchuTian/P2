# =============================================================================
# 1. IMPORT LIBRARIES
# =============================================================================
import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go

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
def generate_health_dashboard(df_input, rules):
    """
    Generates a Plotly figure that acts as a health dashboard based on user inputs.
    """
    dashboard_data = []
    
    # Iterate through rules to evaluate each user input
    for rule in rules:
        feature = rule['feature']
        value = df_input.iloc[0][feature]
        
        # Check if the condition is met
        if rule['condition'](value):
            status = 'Risk' if rule['type'] == 'risk' else 'Healthy'
            color = '#ff6b6b' if status == 'Risk' else '#68d391'
            icon = '⚠️' if status == 'Risk' else '✅'
            description = rule['text'].split(',')[0] # Get the first part of the rule text for the label
            dashboard_data.append({'Category': rule.get('category', 'Lifestyle'), 'Habit': description, 'Status': icon, 'Color': color})
            
    if not dashboard_data:
        return None

    df_plot = pd.DataFrame(dashboard_data)

    # Create the Plotly figure
    fig = go.Figure()

    for i, row in df_plot.iterrows():
        fig.add_trace(go.Scatter(
            x=[i], y=[1],
            mode='markers+text',
            marker=dict(color=row['Color'], size=20, symbol='circle'),
            text=row['Status'],
            textfont=dict(size=14, color='white'),
            hoverinfo='text',
            hovertext=row['Habit'],
            name=row['Habit']
        ))

    fig.update_layout(
        title='<b>Your Personalized Health Habits Dashboard</b>',
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(len(df_plot))),
            ticktext=[f"<b>{row['Habit']}</b>" for i, row in df_plot.iterrows()],
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            range=[-1, 3] # Provide some vertical space
        ),
        showlegend=False,
        height=200,
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

# =============================================================================
# 4. PAGE CONFIGURATION AND SESSION STATE INITIALIZATION
# =============================================================================
st.set_page_config(page_title="Obesity Risk Predictor", page_icon="🍔", layout="wide") # Use wide layout

if 'prediction_label' not in st.session_state:
    st.session_state.prediction_label = None
if 'explanation_text' not in st.session_state:
    st.session_state.explanation_text = None
if 'health_dashboard' not in st.session_state:
    st.session_state.health_dashboard = None
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
        st.session_state.health_dashboard = None


# =============================================================================
# 7. DISPLAY RESULTS AND LOCAL (TEXT) EXPLANATION
# =============================================================================
with explanation_col:
    st.markdown('<div class="section-title">💡 Prediction & Explanation</div>', unsafe_allow_html=True)
    if st.session_state.prediction_label:
        st.success(f"✅ Your predicted obesity category is: **{st.session_state.prediction_label}**")

        if st.button("📊 Explain My Prediction", use_container_width=True):
            # This is the "narrative engine" that generates human-readable text.
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
            
            with st.spinner('Analyzing your health profile...'):
                # --- NEW: Generate dashboard and text simultaneously ---
                st.session_state.health_dashboard = generate_health_dashboard(st.session_state.df_input_original, narrative_rules)
                
                # --- Text generation logic (can be simplified or kept as is) ---
                pred_label = st.session_state.prediction_label
                age_val = st.session_state.df_input_original['Age'].iloc[0]
                weight_val = st.session_state.df_input_original['Weight'].iloc[0]
                
                opening_statement = f"Based on the information you provided, the model predicted your risk category as **{pred_label}**.\n\n"
                opening_statement += f"For an individual of **{age_val:.0f} years old** with a weight of **{weight_val:.0f} kg**, the following habits are the most noteworthy for your health profile:"
                
                risk_narratives = [rule['text'] for rule in narrative_rules if rule['type'] == 'risk' and rule['condition'](st.session_state.df_input_original.iloc[0][rule['feature']])]
                protective_narratives = [rule['text'] for rule in narrative_rules if rule['type'] == 'protective' and rule['condition'](st.session_state.df_input_original.iloc[0][rule['feature']])]
                
                final_explanation = opening_statement
                if risk_narratives:
                    final_explanation += "\n\n**Areas for Improvement (Risk Factors):**\n" + "".join([f"- {text}\n" for text in risk_narratives])
                if protective_narratives:
                    final_explanation += "\n**Positive Habits to Maintain:**\n" + "".join([f"- {text}\n" for text in protective_narratives])
                
                st.session_state.explanation_text = final_explanation

        # --- NEW DISPLAY LOGIC ---
        # Display the dashboard first, then the text.
        if st.session_state.health_dashboard:
            st.plotly_chart(st.session_state.health_dashboard, use_container_width=True)
        
        if st.session_state.explanation_text:
            st.info(st.session_state.explanation_text)
            
    else:
        st.info("Please input your data and click 'Predict' to see the results.")
            

# =============================================================================
# 9. FOOTER
# =============================================================================
st.markdown('<div class="footer">Made with ❤️ by Your AI Assistant</div>', unsafe_allow_html=True)
