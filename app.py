import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import torch

from dataset_generator import generate_synthetic_weather_data
from weather_api import geocode_city, fetch_live_weather
from model_trainer import train_and_save_models, PyTorchWeatherNN

# -----------------------------------------------------------------------------
# 1. Page Config & Google Search Console Verification Meta Tag
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Weather Intelligence | Real-Time Global Forecast & Deep Learning",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Google Search Console Verification Meta Tag Injection
st.markdown('<meta name="google-site-verification" content="googleeb7948709320c598" />', unsafe_allow_html=True)

# Inject Custom Styling
css_file = os.path.join(os.path.dirname(__file__), 'styles.css')
if os.path.exists(css_file):
    with open(css_file) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. Resource Caching & Model Loading (with Auto Environment Compatibility)
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data')

@st.cache_resource
def load_all_artifacts():
    scaler_path = os.path.join(MODELS_DIR, 'scaler.joblib')
    if not os.path.exists(scaler_path):
        train_and_save_models()
        
    try:
        scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.joblib'))
        label_encoder = joblib.load(os.path.join(MODELS_DIR, 'label_encoder.joblib'))
        rf_clf = joblib.load(os.path.join(MODELS_DIR, 'rf_classifier.joblib'))
        gb_reg = joblib.load(os.path.join(MODELS_DIR, 'gb_regressor.joblib'))
        history = joblib.load(os.path.join(MODELS_DIR, 'dl_history.joblib'))
    except Exception as err:
        print(f"Joblib load notice ({err}). Retraining models for local environment compatibility...")
        train_and_save_models()
        scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.joblib'))
        label_encoder = joblib.load(os.path.join(MODELS_DIR, 'label_encoder.joblib'))
        rf_clf = joblib.load(os.path.join(MODELS_DIR, 'rf_classifier.joblib'))
        gb_reg = joblib.load(os.path.join(MODELS_DIR, 'gb_regressor.joblib'))
        history = joblib.load(os.path.join(MODELS_DIR, 'dl_history.joblib'))
    
    num_classes = len(label_encoder.classes_)
    nn_model = PyTorchWeatherNN(input_dim=8, num_classes=num_classes)
    pth_path = os.path.join(MODELS_DIR, 'pytorch_weather_net.pth')
    nn_model.load_state_dict(torch.load(pth_path, weights_only=True))
    nn_model.eval()
    
    return scaler, label_encoder, rf_clf, gb_reg, nn_model, history

try:
    scaler, label_encoder, rf_clf, gb_reg, nn_model, dl_history = load_all_artifacts()
except Exception as e:
    print(f"Fallback training models: {e}")
    train_and_save_models()
    scaler, label_encoder, rf_clf, gb_reg, nn_model, dl_history = load_all_artifacts()

CONDITION_EMOJIS = {
    'Sunny': '☀️',
    'Cloudy': '⛅',
    'Rainy': '🌧️',
    'Snowy': '❄️',
    'Stormy': '🌩️'
}

# -----------------------------------------------------------------------------
# 3. Hero Header & Clean Sidebar
# -----------------------------------------------------------------------------
st.markdown("""
    <div class="hero-container">
        <div class="hero-title">🌤️ Weather Intelligence AI</div>
        <div class="hero-subtitle">Instant Real-Time Global Telemetry & Predictive AI Forecasts</div>
    </div>
""", unsafe_allow_html=True)

st.sidebar.markdown("## 🌤️ Weather Console")
st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Quick Controls")
unit_temp = st.sidebar.radio("Temperature Unit:", ["Celsius (°C)", "Fahrenheit (°F)"])
st.sidebar.markdown("---")

# -----------------------------------------------------------------------------
# 4. Main Tabs Navigation
# -----------------------------------------------------------------------------
tab_live, tab_sim, tab_eval = st.tabs([
    "🌍 Live City Forecast",
    "🎛️ Weather Simulator",
    "📈 Confidence Metrics"
])

def convert_temp(val_c):
    if "Fahrenheit" in unit_temp:
        return (val_c * 9/5) + 32, "°F"
    return val_c, "°C"

# =============================================================================
# TAB 1: Live City Weather
# =============================================================================
with tab_live:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 🌍 Search Global City Weather")
    
    col_input, col_btn = st.columns([4, 1])
    with col_input:
        city_name = st.text_input("City Search", value="London", label_visibility="collapsed", placeholder="Enter any city... (e.g. New York, Tokyo, Paris, Delhi, Sydney)")
    with col_btn:
        st.write("")
        btn_search = st.button("🔍 Get Forecast", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    if city_name:
        city_geo = geocode_city(city_name)
        if city_geo:
            weather_data = fetch_live_weather(city_geo["latitude"], city_geo["longitude"])
            if weather_data:
                temp_disp, temp_unit = convert_temp(weather_data['temperature_c'])
                
                st.markdown(f"## 📍 **{city_geo['name']}, {city_geo['country']}**")
                
                # Glass Tile Stat Metrics
                st.markdown(f"""
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; margin: 20px 0;">
                        <div class="stat-tile">
                            <div class="stat-icon">🌡️</div>
                            <div class="stat-label">Temperature</div>
                            <div class="stat-value">{temp_disp:.1f} {temp_unit}</div>
                        </div>
                        <div class="stat-tile">
                            <div class="stat-icon">💧</div>
                            <div class="stat-label">Humidity</div>
                            <div class="stat-value">{weather_data['humidity_pct']} %</div>
                        </div>
                        <div class="stat-tile">
                            <div class="stat-icon">⏲️</div>
                            <div class="stat-label">Pressure</div>
                            <div class="stat-value">{weather_data['pressure_hpa']} hPa</div>
                        </div>
                        <div class="stat-tile">
                            <div class="stat-icon">💨</div>
                            <div class="stat-label">Wind Speed</div>
                            <div class="stat-value">{weather_data['wind_speed_kmh']} km/h</div>
                        </div>
                        <div class="stat-tile">
                            <div class="stat-icon">☁️</div>
                            <div class="stat-label">Cloud Cover</div>
                            <div class="stat-value">{weather_data['cloud_cover_pct']} %</div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                # Model Inference
                current_month = pd.Timestamp.now().month
                current_hour = pd.Timestamp.now().hour
                uv_est = 5.0 if 8 <= current_hour <= 17 else 0.0
                
                raw_features = np.array([[
                    current_month, current_hour,
                    weather_data['temperature_c'],
                    weather_data['humidity_pct'],
                    weather_data['pressure_hpa'],
                    weather_data['wind_speed_kmh'],
                    weather_data['cloud_cover_pct'],
                    uv_est
                ]])
                
                scaled_features = scaler.transform(raw_features)
                
                # Deep Learning Inference
                with torch.no_grad():
                    t_x = torch.tensor(scaled_features, dtype=torch.float32)
                    logits, reg_temp = nn_model(t_x)
                    probs = torch.softmax(logits, dim=1).numpy()[0]
                    predicted_idx = torch.argmax(logits, dim=1).item()
                    dl_condition = label_encoder.inverse_transform([predicted_idx])[0]
                    dl_next_temp = reg_temp.item()
                    
                # ML Inference
                rf_pred_idx = rf_clf.predict(scaled_features)[0]
                rf_condition = label_encoder.inverse_transform([rf_pred_idx])[0]
                gb_next_temp = gb_reg.predict(scaled_features)[0]
                
                dl_next_disp, _ = convert_temp(dl_next_temp)
                gb_next_disp, _ = convert_temp(gb_next_temp)
                
                st.markdown("### 🤖 Dual-Engine AI Predictions")
                
                col_dl, col_ml = st.columns(2)
                
                with col_dl:
                    st.markdown(f"""
                        <div class="glass-card">
                            <span class="ai-badge badge-pytorch">🔥 Deep Learning Model</span>
                            <div class="pred-condition">
                                <span>{CONDITION_EMOJIS.get(dl_condition, '⛅')}</span>
                                <span>{dl_condition}</span>
                            </div>
                            <p style="color: #94a3b8; font-size: 0.9rem; margin-bottom: 12px;">Confidence: <strong>{probs[predicted_idx]*100:.1f}%</strong></p>
                            <div class="ai-prediction-card">
                                <div style="font-size: 0.8rem; text-transform: uppercase; color: #94a3b8; letter-spacing: 0.05em;">Upcoming Temp Forecast</div>
                                <div class="pred-temp">🌡️ {dl_next_disp:.1f} {temp_unit}</div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                with col_ml:
                    st.markdown(f"""
                        <div class="glass-card">
                            <span class="ai-badge badge-ml">🌲 Machine Learning Model</span>
                            <div class="pred-condition">
                                <span>{CONDITION_EMOJIS.get(rf_condition, '⛅')}</span>
                                <span>{rf_condition}</span>
                            </div>
                            <p style="color: #94a3b8; font-size: 0.9rem; margin-bottom: 12px;">Random Forest Predictor</p>
                            <div class="ai-prediction-card" style="border-left-color: #818cf8;">
                                <div style="font-size: 0.8rem; text-transform: uppercase; color: #94a3b8; letter-spacing: 0.05em;">Upcoming Temp Forecast</div>
                                <div class="pred-temp" style="color: #818cf8;">🌡️ {gb_next_disp:.1f} {temp_unit}</div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Plotly 7-Day Forecast Chart
                if "daily" in weather_data and "time" in weather_data["daily"]:
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    st.markdown("### 📅 7-Day Temperature Trend")
                    daily_temps_max = [convert_temp(x)[0] for x in weather_data["daily"]["temperature_2m_max"]]
                    daily_temps_min = [convert_temp(x)[0] for x in weather_data["daily"]["temperature_2m_min"]]
                    
                    daily_df = pd.DataFrame({
                        "Date": weather_data["daily"]["time"],
                        f"Max Temp ({temp_unit})": daily_temps_max,
                        f"Min Temp ({temp_unit})": daily_temps_min
                    })
                    
                    fig = px.line(daily_df, x="Date", y=[f"Max Temp ({temp_unit})", f"Min Temp ({temp_unit})"],
                                  markers=True, template="plotly_dark",
                                  color_discrete_sequence=["#ef4444", "#38bdf8"])
                    fig.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(family="Plus Jakarta Sans", color="#94a3b8"),
                        margin=dict(l=10, r=10, t=20, b=10),
                        height=320
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.error("Failed to fetch live weather details for the specified city.")
        else:
            st.warning(f"Could not locate '{city_name}'. Please check the city spelling.")

# =============================================================================
# TAB 2: Custom Weather Simulator
# =============================================================================
with tab_sim:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 🎛️ Weather Telemetry Simulator")
    st.write("Drag controls to simulate custom climate conditions and generate real-time AI weather forecasts.")
    
    col_s1, col_s2, col_s3 = st.columns(3)
    
    with col_s1:
        sim_month = st.slider("Month of Year", 1, 12, 7)
        sim_hour = st.slider("Hour of Day (24h)", 0, 23, 14)
        sim_temp = st.slider("Temperature (°C)", -20.0, 45.0, 26.0, step=0.5)
        
    with col_s2:
        sim_humidity = st.slider("Relative Humidity (%)", 10.0, 100.0, 65.0, step=1.0)
        sim_pressure = st.slider("Atmospheric Pressure (hPa)", 970.0, 1040.0, 1012.0, step=0.5)
        sim_wind = st.slider("Wind Speed (km/h)", 0.0, 100.0, 18.0, step=1.0)
        
    with col_s3:
        sim_cloud = st.slider("Cloud Cover (%)", 0.0, 100.0, 45.0, step=1.0)
        sim_uv = st.slider("UV Index", 0.0, 12.0, 6.5, step=0.1)
    st.markdown('</div>', unsafe_allow_html=True)
        
    raw_sim = np.array([[sim_month, sim_hour, sim_temp, sim_humidity, sim_pressure, sim_wind, sim_cloud, sim_uv]])
    scaled_sim = scaler.transform(raw_sim)
    
    # Deep Learning Inference
    with torch.no_grad():
        t_sim = torch.tensor(scaled_sim, dtype=torch.float32)
        logits_sim, reg_sim = nn_model(t_sim)
        probs_sim = torch.softmax(logits_sim, dim=1).numpy()[0]
        pred_idx_sim = torch.argmax(logits_sim, dim=1).item()
        dl_cond_sim = label_encoder.inverse_transform([pred_idx_sim])[0]
        dl_next_sim = reg_sim.item()
        
    # Machine Learning Inference
    rf_sim_idx = rf_clf.predict(scaled_sim)[0]
    rf_cond_sim = label_encoder.inverse_transform([rf_sim_idx])[0]
    gb_next_sim = gb_reg.predict(scaled_sim)[0]
    
    dl_sim_disp, temp_unit = convert_temp(dl_next_sim)
    gb_sim_disp, _ = convert_temp(gb_next_sim)
    
    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        st.markdown(f"""
            <div class="glass-card">
                <span class="ai-badge badge-pytorch">🔥 Deep Learning Model</span>
                <div class="pred-condition">
                    <span>{CONDITION_EMOJIS.get(dl_cond_sim, '⛅')}</span>
                    <span>{dl_cond_sim}</span>
                </div>
                <p style="color: #94a3b8; font-size: 0.9rem;">Confidence: <strong>{probs_sim[pred_idx_sim]*100:.1f}%</strong></p>
                <div class="ai-prediction-card">
                    <div style="font-size: 0.8rem; text-transform: uppercase; color: #94a3b8; letter-spacing: 0.05em;">Predicted Next Temp</div>
                    <div class="pred-temp">🌡️ {dl_sim_disp:.1f} {temp_unit}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
    with res_col2:
        st.markdown(f"""
            <div class="glass-card">
                <span class="ai-badge badge-ml">🌲 Machine Learning Model</span>
                <div class="pred-condition">
                    <span>{CONDITION_EMOJIS.get(rf_cond_sim, '⛅')}</span>
                    <span>{rf_cond_sim}</span>
                </div>
                <p style="color: #94a3b8; font-size: 0.9rem;">Random Forest Ensemble</p>
                <div class="ai-prediction-card" style="border-left-color: #818cf8;">
                    <div style="font-size: 0.8rem; text-transform: uppercase; color: #94a3b8; letter-spacing: 0.05em;">Predicted Next Temp</div>
                    <div class="pred-temp" style="color: #818cf8;">🌡️ {gb_sim_disp:.1f} {temp_unit}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📊 Class Probability Breakdown")
    classes = label_encoder.classes_
    prob_df = pd.DataFrame({"Condition": classes, "Probability": probs_sim})
    
    fig_prob = px.bar(prob_df, x="Condition", y="Probability", color="Condition",
                      text_auto=".1%", template="plotly_dark",
                      color_discrete_sequence=px.colors.qualitative.Vivid)
    fig_prob.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Plus Jakarta Sans", color="#94a3b8"),
        height=320
    )
    st.plotly_chart(fig_prob, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# TAB 3: Confidence Metrics
# =============================================================================
with tab_eval:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📈 Model Optimization & Accuracy Metrics")
    
    col_m1, col_m2 = st.columns(2)
    hist_df = pd.DataFrame(dl_history)
    
    with col_m1:
        fig_loss = px.line(hist_df, x="epoch", y="accuracy",
                           title="Model Accuracy Growth",
                           template="plotly_dark",
                           color_discrete_sequence=["#38bdf8"])
        fig_loss.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=320)
        st.plotly_chart(fig_loss, use_container_width=True)
        
    with col_m2:
        fig_acc = px.line(hist_df, x="epoch", y="loss",
                          title="Loss Curve Optimization",
                          template="plotly_dark", color_discrete_sequence=["#34d399"])
        fig_acc.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=320)
        st.plotly_chart(fig_acc, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
