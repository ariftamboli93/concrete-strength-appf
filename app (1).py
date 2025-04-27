import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Page Config
st.set_page_config(page_title="Concrete Strength Predictor", layout="centered", page_icon="🧱")

# Custom Dark Theme Styling
st.markdown("""
    <style>
        html, body, [class*="css"]  {
            font-family: 'Segoe UI', sans-serif;
            background-color: #000000;
            color: #ffffff;
        }
        .title {
            font-size: 3em;
            text-align: center;
            color: #00ffff;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .info-card {
            background-color: #1c1c1c;
            padding: 20px;
            border-left: 5px solid #00cccc;
            border-radius: 10px;
            color: #ffffff;
            font-size: 18px;
        }
        .subheader {
            font-size: 1.5em;
            margin-top: 30px;
            color: #00ccff;
            font-weight: 600;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="title">🧪 Concrete Compressive Strength Predictor</div>', unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/esvs2202/Concrete-Compressive-Strength-Prediction/refs/heads/main/dataset/concrete_data.csv"
    return pd.read_csv(url)

df = load_data()
X = df.drop('concrete_compressive_strength', axis=1)
y = df['concrete_compressive_strength']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Sidebar Inputs (In Kg instead of Kg/m³)
st.sidebar.header("🔧 Input Mix Details")
cement = st.sidebar.number_input("Cement (kg)", min_value=0.0, value=540.0)  # In kg
slag = st.sidebar.number_input("Blast Furnace Slag (kg)", min_value=0.0, value=0.0)  # In kg
fly_ash = st.sidebar.number_input("Fly Ash (kg)", min_value=0.0, value=0.0)  # In kg
water = st.sidebar.number_input("Water (kg)", min_value=0.0, value=162.0)  # In kg
superplasticizer = st.sidebar.number_input("Superplasticizer (kg)", min_value=0.0, value=2.5)  # In kg
coarse_agg = st.sidebar.number_input("Coarse Aggregate (kg)", min_value=0.0, value=1040.0)  # In kg
fine_agg = st.sidebar.number_input("Fine Aggregate (kg)", min_value=0.0, value=676.0)  # In kg
age = st.sidebar.number_input("Age (days)", min_value=1, value=28)

# Predict Button
if st.sidebar.button("🧮 Predict Strength"):
    input_data = np.array([[cement, slag, fly_ash, water, superplasticizer, coarse_agg, fine_agg, age]])
    prediction = model.predict(input_data)
    st.markdown(f'<div class="info-card">✅ <strong>Predicted Strength:</strong> {prediction[0]:.2f} MPa</div>', unsafe_allow_html=True)

    # Model Predictions for test set
    y_pred = model.predict(X_test)

    # Actual vs Predicted
    st.markdown('<div class="subheader">📊 Actual vs Predicted</div>', unsafe_allow_html=True)
    fig1, ax1 = plt.subplots(facecolor='black')
    ax1.scatter(y_test, y_pred, color='#00ffff', alpha=0.6, label="Data Points")
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label="Perfect Prediction")
    ax1.set_facecolor('black')
    ax1.set_xlabel("Actual Strength", color='white')
    ax1.set_ylabel("Predicted Strength", color='white')
    ax1.set_title("Actual vs Predicted Strength", color='white')
    ax1.tick_params(colors='white')
    ax1.legend()
    st.pyplot(fig1)

    # Residual Plot
    st.markdown('<div class="subheader">📉 Residual Plot</div>', unsafe_allow_html=True)
    residuals = y_test - y_pred
    fig2, ax2 = plt.subplots(facecolor='black')
    ax2.scatter(y_pred, residuals, alpha=0.6, color='#ff80ab')
    ax2.hlines(0, y_pred.min(), y_pred.max(), colors='gray', linestyles='dashed')
    ax2.set_facecolor('black')
    ax2.set_xlabel("Predicted Strength", color='white')
    ax2.set_ylabel("Residuals", color='white')
    ax2.set_title("Residuals vs Prediction", color='white')
    ax2.tick_params(colors='white')
    st.pyplot(fig2)

    # Coefficients
    st.markdown('<div class="subheader">📌 Feature Importance</div>', unsafe_allow_html=True)
    coef_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_})
    coef_df = coef_df.sort_values(by='Coefficient', key=abs)
    fig3, ax3 = plt.subplots(facecolor='black')
    ax3.barh(coef_df['Feature'], coef_df['Coefficient'], color='#00e676')
    ax3.set_facecolor('black')
    ax3.set_title("Model Coefficients", color='white')
    ax3.set_xlabel("Coefficient Value", color='white')
    ax3.tick_params(colors='white')
    st.pyplot(fig3)

    # Metrics
    st.markdown('<div class="subheader">📈 Model Performance</div>', unsafe_allow_html=True)
    st.success(f"**R² Score:** {r2_score(y_test, y_pred):.2f}")
    st.info(f"**Mean Squared Error:** {mean_squared_error(y_test, y_pred):.2f}")
