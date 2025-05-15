import streamlit as st
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import folium
from streamlit_folium import st_folium

# Set page config
st.set_page_config(page_title="House Price Predictor", layout="centered")

# Apply enhanced CSS styling including new info-box style
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Roboto', sans-serif;
        background-color: #0d1117;
        color: white;
    }

    .stApp {
        background-image: url("https://images.unsplash.com/photo-1600585154340-be6161a56a0c");
        background-size: cover;
        background-attachment: fixed;
    }

    .main > div {
        background-color: rgba(0, 0, 0, 0.75);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0px 0px 20px rgba(255, 255, 255, 0.05);
    }

    h1, h2, h3 {
        text-align: center;
        font-weight: 700;
        color: #ffffff;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.85);
    }

    .headline-block {
        background: rgba(0, 0, 0, 0.75);
        border-radius: 15px;
        padding: 1.5rem;
        margin-bottom: 2rem;
        backdrop-filter: blur(4px);
        box-shadow: 0 4px 20px rgba(0,0,0,0.6);
        color: #fff;
        text-align: center;
    }

    .headline-block h1, .headline-block h3 {
        color: #ffffff;
        text-shadow: 3px 3px 10px rgba(0,0,0,0.9);
    }

    section[data-testid="stSidebar"] {
        background-color: rgba(20, 20, 30, 0.95);
        color: white;
        padding: 20px;
        border-right: 2px solid #444;
        box-shadow: 5px 0px 15px rgba(0, 0, 0, 0.5);
    }

    .stSlider label, .stSelectbox label, .stNumberInput label {
        color: white !important;
        font-weight: bold;
    }

    button[kind="primary"] {
        background-color: #f39c12 !important;
        color: white !important;
        border: none;
        padding: 0.6rem 1.2rem;
        font-size: 1.1rem;
        font-weight: bold;
        border-radius: 8px;
        box-shadow: 0px 5px 15px rgba(243, 156, 18, 0.4);
        transition: all 0.3s ease;
    }

    button[kind="primary"]:hover {
        background-color: #e67e22 !important;
        box-shadow: 0px 8px 20px rgba(230, 126, 34, 0.5);
    }

    .stButton > button {
        width: 100%;
        margin-bottom: 0.5rem;
    }

    .st-expanderHeader {
        font-weight: bold;
        color: #f1f1f1 !important;
    }

    .stInfo {
        background-color: rgba(255, 255, 255, 0.1);
        padding: 0.75rem;
        border-left: 4px solid #f39c12;
        border-radius: 8px;
        margin-bottom: 1rem;
    }

    /* New style for info box backgrounds */
    .info-box {
        background-color: rgba(0, 0, 0, 0.75);
        border-radius: 15px;
        padding: 15px;
        margin-bottom: 20px;
        box-shadow: 0 0 15px rgba(0,0,0,0.7);
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    try:
        model = joblib.load("xgb_model_fixed.pkl")
        scaler = joblib.load("scaler_fixed.pkl")
        features_used = joblib.load("feature_names.pkl")
    except:
        data = fetch_california_housing()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = data.target * 100000
        features_used = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup"]
        X = X[features_used]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
        model.fit(X_scaled, y)
        joblib.dump(model, "xgb_model_fixed.pkl")
        joblib.dump(scaler, "scaler_fixed.pkl")
        joblib.dump(features_used, "feature_names.pkl")
    return model, scaler, features_used

@st.cache_data
def get_housing_data():
    data = fetch_california_housing()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["Target"] = data.target * 100000
    return df

@st.cache_data
def evaluate_model():
    data = fetch_california_housing()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target * 100000
    features_used = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup"]
    X = X[features_used]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = joblib.load("xgb_model_fixed.pkl")
    y_pred = model.predict(X_test_scaled)
    return mean_absolute_error(y_test, y_pred), np.sqrt(mean_squared_error(y_test, y_pred)), r2_score(y_test, y_pred)

model, scaler, features_used = load_model()
housing_data = get_housing_data()
mae, rmse, r2 = evaluate_model()

# Header section
st.markdown("""
<div class="headline-block">
    <h1>🏠 House Price Prediction App</h1>
    <h3>Predict house prices in California based on lifestyle-friendly inputs</h3>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("Enter Home Features:")
user_input = {}
user_input["MedInc"] = st.sidebar.slider("Median Household Income ($)", 10000, 150000, 50000, step=5000) / 10000
user_input["HouseAge"] = st.sidebar.selectbox("Age of the House (years)", list(range(1, 52)))
user_input["AveRooms"] = st.sidebar.selectbox("Number of Rooms", list(range(1, 11)))
user_input["AveBedrms"] = st.sidebar.selectbox("Number of Bedrooms", list(range(1, 6)))
user_input["Population"] = st.sidebar.slider("Neighborhood Population", 100, 5000, 1000, step=100)
user_input["AveOccup"] = st.sidebar.slider("Average Occupancy (people/household)", 1.0, 10.0, 3.0, step=0.1)
location = st.sidebar.selectbox("Preferred Area", ["Suburban", "Midtown", "Outer"])
location_mapping = {
    "Suburban": (34.0, -118.5),
    "Midtown": (37.5, -122.0),
    "Outer": (36.0, -119.0)
}
user_input["Latitude"], user_input["Longitude"] = location_mapping[location]

# Prediction logic
if st.button("Predict Price 💰"):
    input_df = pd.DataFrame([{k: user_input[k] for k in features_used}])
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    st.session_state["prediction"] = prediction
    st.session_state["lat"] = user_input["Latitude"]
    st.session_state["lon"] = user_input["Longitude"]
    st.session_state["input_df"] = input_df
    st.session_state["user_input"] = user_input
    st.session_state["show_map"] = False

    def find_closest_match(user_input, dataset):
        df_filtered = dataset.copy()
        df_filtered["distance"] = ((df_filtered[features_used] - input_df.values[0])**2).sum(axis=1)
        return df_filtered.loc[df_filtered["distance"].idxmin()]

    closest = find_closest_match(user_input, housing_data)
    actual = closest["Target"]
    error = abs(prediction - actual)
    error_pct = (error / actual) * 100
    st.session_state["error_pct"] = error_pct
    st.session_state["actual_price"] = actual
    st.session_state["error_amount"] = error

# Results
if "prediction" in st.session_state:
    if st.session_state["error_pct"] > 30:
        st.warning("❌ No house found within your range of income and preferred specifications. Please adjust your input.")
    else:
        st.markdown(f"""
        <div class="info-box">
        🏡 Estimated House Price: <b>${st.session_state['prediction']:,.2f}</b>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-box">
        ### 📊 Comparison with Real Data
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="info-box">
        <b>Closest Real House Price:</b> ${st.session_state['actual_price']:,.2f}<br>
        <b>Prediction Error:</b> ${st.session_state['error_amount']:,.2f} ({st.session_state['error_pct']:.2f}%)
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-box">
        ### 📈 Model Performance on Test Set
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="info-box">
        <b>Mean Absolute Error (MAE):</b> ${mae:,.2f}<br>
        <b>Root Mean Squared Error (RMSE):</b> ${rmse:,.2f}<br>
        <b>R² Score:</b> {r2:.4f}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-box">
        ### ⭐ Neighborhood Reviews
        </div>
        """, unsafe_allow_html=True)

        st.info("🗣️ *\u201cLovely quiet neighborhood with great schools. Safe and friendly community!\u201d* – ★★★★★")
        st.info("🗣️ *\u201cDecent place but traffic can be a bit much during peak hours.\u201d* – ★★★☆☆")
        st.info("🗣️ *\u201cAffordable homes and great grocery options nearby.\u201d* – ★★★★☆")

        with st.expander("📊 See Input Data"):
            st.dataframe(pd.DataFrame([st.session_state["user_input"]]))

# Map
if st.button("Show Location on Map 🗺️"):
    st.session_state["show_map"] = True

if st.session_state.get("show_map", False):
    if "lat" in st.session_state and "lon" in st.session_state:
        m = folium.Map(location=[st.session_state["lat"], st.session_state["lon"]], zoom_start=8)
        folium.Marker(
            location=[st.session_state["lat"], st.session_state["lon"]],
            popup="Predicted House Location",
            icon=folium.Icon(color="green", icon="flag")
        ).add_to(m)
        st_folium(m, width=500, height=350)
    else:
        st.info("🔮 Please predict a price first to view map.")
