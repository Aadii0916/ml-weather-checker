# 🌐 Guide: Adding Custom APIs & Deploying Your Streamlit App Live

This guide explains how to integrate custom Weather API keys (like OpenWeatherMap) and how to deploy your app live on the internet for free so anyone can use it.

---

## 🔑 Part 1: How to Add Custom Weather APIs

Currently, your app uses **Open-Meteo API** which is 100% free and requires **no API key**. If you want to use **OpenWeatherMap**:

### Step 1: Get a free API Key
1. Sign up at [openweathermap.org](https://openweathermap.org/api).
2. Get your free API Key from your account dashboard.

### Step 2: Add API Key safely in Streamlit Secrets
Create a file named `.streamlit/secrets.toml` inside your project folder:

```toml
# C:\Users\himan\.gemini\antigravity\scratch\ml_weather_checker\.streamlit\secrets.toml
OPENWEATHER_API_KEY = "your_actual_api_key_here"
```

### Step 3: Use the key in `weather_api.py`
In `weather_api.py`:
```python
import streamlit as st
import requests

api_key = st.secrets.get("OPENWEATHER_API_KEY", "")
url = f"https://api.openweathermap.org/data/2.5/weather?q={city_name}&appid={api_key}&units=metric"
```

---

## 🚀 Part 2: How to Make Your App LIVE Online (Free Deployment)

### Method 1: Streamlit Community Cloud (Recommended & 100% Free)
This is the easiest way to give your app a permanent public URL (e.g. `https://your-weather-app.streamlit.app`).

1. **Upload your code to GitHub**:
   - Create a new free repository on [github.com](https://github.com).
   - Upload the files in `C:\Users\himan\.gemini\antigravity\scratch\ml_weather_checker` to your repository.

2. **Deploy on Streamlit Community Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
   - Click **New app**.
   - Select your GitHub repository, set Main file path to `app.py`, and click **Deploy**!
   - In 2 minutes, your app will be live on the internet!

---

### Method 2: Instant Public Link with Localtunnel (No Upload Needed!)
If you want to share a live link right now without uploading to GitHub:

1. Open your terminal in `C:\Users\himan\.gemini\antigravity\scratch\ml_weather_checker`.
2. Run this command:
   ```bash
   npx localtunnel --port 8502
   ```
3. It will give you a public URL (e.g. `https://fuzzy-cloud-99.loca.lt`) that anyone on any phone or laptop can open while your app is running!

---

### Method 3: Share on local Wi-Fi
Anyone connected to the same Wi-Fi network as your computer can open:
`http://<YOUR_COMPUTER_IP>:8502`
