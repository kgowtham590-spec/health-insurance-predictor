# ============================================================
# Health Insurance Premium Predictor - Streamlit Web App
# Run with: streamlit run app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Page config
st.set_page_config(
    page_title="Health Insurance Premium Predictor",
    page_icon="🏥",
    layout="centered"
)

# ---- Train model on load ----
@st.cache_resource
def train_model():
    df = pd.read_csv('insurance.csv')
    df_processed = df.copy()
    le = LabelEncoder()
    df_processed['sex'] = le.fit_transform(df_processed['sex'])
    df_processed['smoker'] = le.fit_transform(df_processed['smoker'])
    df_processed['region'] = le.fit_transform(df_processed['region'])
    X = df_processed.drop('charges', axis=1)
    y = df_processed['charges']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

model = train_model()

# ---- UI ----
st.title("🏥 Health Insurance Premium Predictor")
st.markdown("*Built with Random Forest | Dataset: Medical Cost Personal Dataset*")
st.markdown("---")

st.subheader("Enter Your Details")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", min_value=18, max_value=65, value=30)
    bmi = st.slider("BMI", min_value=15.0, max_value=55.0, value=25.0, step=0.1)
    children = st.selectbox("Number of Children", [0, 1, 2, 3, 4, 5])

with col2:
    sex = st.selectbox("Sex", ["Male", "Female"])
    smoker = st.selectbox("Smoker?", ["No", "Yes"])
    region = st.selectbox("Region", ["Northeast", "Northwest", "Southeast", "Southwest"])

# Encode inputs
sex_enc = 1 if sex == "Male" else 0
smoker_enc = 1 if smoker == "Yes" else 0
region_map = {"Northeast": 0, "Northwest": 1, "Southeast": 2, "Southwest": 3}
region_enc = region_map[region]

input_data = pd.DataFrame({
    'age': [age],
    'sex': [sex_enc],
    'bmi': [bmi],
    'children': [children],
    'smoker': [smoker_enc],
    'region': [region_enc]
})

st.markdown("---")

if st.button("🔍 Predict My Premium", use_container_width=True):
    prediction = model.predict(input_data)[0]

    st.success(f"### 💰 Estimated Annual Premium: **${prediction:,.2f}**")
    st.info(f"Monthly: **${prediction/12:,.2f}**")

    # BMI category
    if bmi < 18.5:
        bmi_cat = "Underweight"
    elif bmi < 25:
        bmi_cat = "Normal"
    elif bmi < 30:
        bmi_cat = "Overweight"
    else:
        bmi_cat = "Obese"

    st.markdown("---")
    st.subheader("📊 Key Insights for Your Profile")

    # Compare smoker vs non-smoker
    non_smoker_input = input_data.copy()
    non_smoker_input['smoker'] = 0
    smoker_input = input_data.copy()
    smoker_input['smoker'] = 1

    non_smoker_pred = model.predict(non_smoker_input)[0]
    smoker_pred = model.predict(smoker_input)[0]
    smoking_cost = smoker_pred - non_smoker_pred

    col3, col4 = st.columns(2)
    with col3:
        st.metric("BMI Category", bmi_cat)
        st.metric("Non-Smoker Premium", f"${non_smoker_pred:,.0f}")
    with col4:
        st.metric("Smoking Premium Impact", f"+${smoking_cost:,.0f}/yr",
                  delta=f"{'You save this!' if smoker == 'No' else 'Extra cost of smoking'}",
                  delta_color="normal" if smoker == "No" else "inverse")
        st.metric("Smoker Premium", f"${smoker_pred:,.0f}")

    # Feature importance bar chart
    st.markdown("---")
    st.subheader("🔍 What Affects Your Premium Most?")
    feature_names = ['Age', 'Sex', 'BMI', 'Children', 'Smoker', 'Region']
    importances = model.feature_importances_

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ['#FF6B6B' if f == 'Smoker' else '#4ECDC4' for f in feature_names]
    bars = ax.barh(feature_names, importances, color=colors)
    ax.set_xlabel('Importance Score')
    ax.set_title('Feature Importance')
    ax.invert_yaxis()
    st.pyplot(fig)

st.markdown("---")
st.caption("🔬 Model: Random Forest Regressor | 📁 Dataset: Kaggle Medical Cost Personal Dataset | Built as ML portfolio project")
