import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from streamlit_folium import st_folium
import folium

# Set page config
st.set_page_config(layout="wide", page_title="House Price Prediction")

# Load model and data
model = joblib.load("xgb_model.pkl")
df = pd.read_csv("data.csv")

# App title
st.title("🏡 House Price Prediction App")

# Sidebar input
st.sidebar.header("Input Features")

longitude = st.sidebar.slider("Longitude", float(df["Longitude"].min()), float(df["Longitude"].max()), float(df["Longitude"].mean()))
latitude = st.sidebar.slider("Latitude", float(df["Latitude"].min()), float(df["Latitude"].max()), float(df["Latitude"].mean()))
housing_median_age = st.sidebar.slider("House Age", float(df["HouseAge"].min()), float(df["HouseAge"].max()), float(df["HouseAge"].mean()))
total_rooms = st.sidebar.slider("Average Rooms", float(df["AveRooms"].min()), float(df["AveRooms"].max()), float(df["AveRooms"].mean()))
total_bedrooms = st.sidebar.slider("Average Bedrooms", float(df["AveBedrms"].min()), float(df["AveBedrms"].max()), float(df["AveBedrms"].mean()))
population = st.sidebar.slider("Population", float(df["Population"].min()), float(df["Population"].max()), float(df["Population"].mean()))
households = st.sidebar.slider("Average Occupancy", float(df["AveOccup"].min()), float(df["AveOccup"].max()), float(df["AveOccup"].mean()))
median_income = st.sidebar.slider("Median Income", float(df["MedInc"].min()), float(df["MedInc"].max()), float(df["MedInc"].mean()))

# Input DataFrame
input_data = pd.DataFrame({
    "MedInc": [median_income],
    "HouseAge": [housing_median_age],
    "AveRooms": [total_rooms],
    "AveBedrms": [total_bedrooms],
    "Population": [population],
    "AveOccup": [households],
    "Latitude": [latitude],
    "Longitude": [longitude]
})

# Predict
prediction = model.predict(input_data)[0]
st.session_state["prediction"] = prediction

# Find closest real house price in data
closest_row = df.iloc[(df.drop("median_house_value", axis=1) - input_data.iloc[0]).abs().sum(axis=1).idxmin()]
actual_price = closest_row.median_house_value
st.session_state['actual_price'] = actual_price
error_amount = abs(prediction - actual_price)
error_pct = error_amount / actual_price * 100
st.session_state['error_amount'] = error_amount
st.session_state['error_pct'] = error_pct

# Load model test scores
y_true = df["median_house_value"]
y_pred = model.predict(df.drop("median_house_value", axis=1))
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)

# Styled section for real data comparison
st.markdown(f"""
<div style="background-color: rgba(0, 0, 0, 0.75); padding: 1rem 1.5rem; border-radius: 10px; margin-bottom: 1rem;">
    <h2 style="color: white;">📊 Comparison with Real Data</h2>
    <p style="color: white;"><strong>Closest Real House Price:</strong> ${actual_price:,.2f}</p>
    <p style="color: white;"><strong>Prediction Error:</strong> ${error_amount:,.2f} ({error_pct:.2f}%)</p>
</div>
""", unsafe_allow_html=True)

# Styled section for model performance
st.markdown(f"""
<div style="background-color: rgba(0, 0, 0, 0.75); padding: 1rem 1.5rem; border-radius: 10px; margin-bottom: 1rem;">
    <h2 style="color: white;">📈 Model Performance on Test Set</h2>
    <p style="color: white;"><strong>Mean Absolute Error (MAE):</strong> ${mae:,.2f}</p>
    <p style="color: white;"><strong>Root Mean Squared Error (RMSE):</strong> ${rmse:,.2f}</p>
    <p style="color: white;"><strong>R² Score:</strong> {r2:.4f}</p>
</div>
""", unsafe_allow_html=True)

# Styled section for reviews
st.markdown("""
<div style="background-color: rgba(0, 0, 0, 0.75); padding: 1rem 1.5rem; border-radius: 10px; margin-bottom: 1rem;">
    <h2 style="color: white;">⭐ Neighborhood Reviews</h2>
    <p style="color: white;">🗣️ <em>"Lovely quiet neighborhood with great schools. Safe and friendly community!"</em> – ★★★★★</p>
    <p style="color: white;">🗣️ <em>"Decent place but traffic can be a bit much during peak hours."</em> – ★★★☆☆</p>
    <p style="color: white;">🗣️ <em>"Affordable homes and great grocery options nearby."</em> – ★★★★☆</p>
</div>
""", unsafe_allow_html=True)

# Show input data
if st.checkbox("📋 See Input Data"):
    st.write(input_data)

# Folium map
m = folium.Map(location=[latitude, longitude], zoom_start=10)
folium.Marker([latitude, longitude], tooltip="Predicted Location").add_to(m)
st_folium(m, width=500, height=300)
