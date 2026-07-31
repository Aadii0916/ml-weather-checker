# 🌤️ AI Weather Checker (Deep Learning & Streamlit)

A complete end-to-end Weather Forecasting & Climate Analytics web application powered by **PyTorch Deep Neural Networks**, **Scikit-Learn Machine Learning**, **Open-Meteo Real-Time Weather API**, and **Streamlit**.

All files and code are completely self-contained in this single project folder:
`C:\Users\himan\.gemini\antigravity\scratch\ml_weather_checker`

---

## 🚀 Quick Start Guide

### Option 1: Double-Click Launcher
Simply double-click `run_app.bat` inside this folder!

### Option 2: Terminal Command
```bash
cd C:\Users\himan\.gemini\antigravity\scratch\ml_weather_checker
streamlit run app.py
```

---

## 🛠️ Project Structure & Architecture

```
ml_weather_checker/
├── app.py                 # Main Streamlit Dashboard Application
├── dataset_generator.py   # Synthesizes 3,000+ realistic meteorological data records
├── model_trainer.py       # Trains PyTorch Multi-Task NN & Scikit-Learn Random Forest
├── weather_api.py         # Open-Meteo live API integration for any global city
├── styles.css             # Glassmorphism dark mode UI styling & custom fonts
├── requirements.txt       # Essential Python libraries (torch, streamlit, scikit-learn, plotly)
├── run_app.bat            # One-click Windows launcher
└── README.md              # Project documentation
```

---

## 🧠 Key Features

1. **🌍 Live City Lookup & Forecast**: Type any city in the world (e.g. London, Tokyo, New York, Delhi) to retrieve real-time weather metrics (temp, humidity, pressure, wind, cloudiness) and feed them directly into your PyTorch Deep Learning model for instant condition & temperature predictions.
2. **🎛️ Interactive Parameter Simulator**: Tweak sliders for temperature, humidity, pressure, UV index, cloud cover, season/hour, and observe live probability distributions across weather states.
3. **📈 PyTorch Multi-Task Deep Neural Network**: Multi-layer perceptron with Batch Normalization and Dropout featuring two output heads:
   - **Classification Head**: Classifies weather condition (Sunny, Cloudy, Rainy, Snowy, Stormy).
   - **Regression Head**: Predicts future temperature (°C).
4. **🌲 Scikit-Learn Benchmarking**: Side-by-side comparison with Random Forest & Gradient Boosting models + Feature Importance charts.
5. **📉 Exploratory Data Analytics (EDA)**: Interactive 3D and 2D Plotly scatter plots, wind speed box plots, and 7-day temperature trends.

---

## 🔒 Safety & File Integrity Guarantee
This project runs entirely inside this isolated directory (`C:\Users\himan\.gemini\antigravity\scratch\ml_weather_checker`). It does **not** alter, delete, or affect any external files or folders on your computer.
