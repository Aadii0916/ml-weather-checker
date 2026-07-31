import numpy as np
import pandas as pd
import os

def generate_synthetic_weather_data(samples=3000, seed=42):
    """
    Generates a realistic synthetic meteorological dataset with physical correlations:
    - Temperature fluctuates based on month and hour.
    - Humidity is inversely related to temperature.
    - Cloud cover and low pressure correlate with rainfall/storminess.
    - Snow occurs at low temperatures (< 2°C).
    """
    np.random.seed(seed)
    
    # Time features
    months = np.random.randint(1, 13, size=samples)
    hours = np.random.randint(0, 24, size=samples)
    
    # Base seasonal temp: Northern hemisphere curve (cold in Jan, warm in July)
    season_factor = np.sin((months - 4) * (2 * np.pi / 12)) * 14
    diurnal_factor = np.sin((hours - 8) * (2 * np.pi / 24)) * 6
    
    base_temp = 12 + season_factor + diurnal_factor
    temperature = base_temp + np.random.normal(0, 4, size=samples)
    
    # Humidity: Inverse to temperature + noise (bounded 15% - 100%)
    humidity = np.clip(85 - (temperature * 1.5) + np.random.normal(0, 10, size=samples), 15, 100)
    
    # Atmospheric pressure (hPa): mean ~1013 hPa
    pressure = 1013 + np.random.normal(0, 12, size=samples)
    
    # Cloud cover (%): correlated with humidity and low pressure
    cloud_cover = np.clip((humidity * 0.7) + (1013 - pressure) * 1.5 + np.random.normal(0, 15, size=samples), 0, 100)
    
    # Wind speed (km/h): higher during low pressure storms
    wind_speed = np.clip(12 + np.abs(1013 - pressure) * 1.2 + np.random.normal(0, 8, size=samples), 0, 90)
    
    # UV Index: proportional to daylight hours, high solar angle, low cloud cover
    is_day = (hours >= 6) & (hours <= 19)
    solar_intensity = np.where(is_day, np.sin((hours - 6) * np.pi / 13), 0)
    uv_index = np.clip(solar_intensity * 10 * (1 - cloud_cover / 120) + np.random.normal(0, 0.5, size=samples), 0, 12)
    uv_index = np.round(uv_index, 1)
    
    # Determine weather condition based on meteorological physics rules + noise
    conditions = []
    for t, h, p, c, w in zip(temperature, humidity, pressure, cloud_cover, wind_speed):
        if c > 70 and h > 75 and p < 1005 and w > 45:
            cond = 'Stormy'
        elif c > 60 and h > 70:
            if t <= 2:
                cond = 'Snowy'
            else:
                cond = 'Rainy'
        elif c > 40:
            cond = 'Cloudy'
        else:
            cond = 'Sunny'
        conditions.append(cond)
        
    # Target regression: Next period temperature (temperature shift + physics trend)
    next_temp = temperature + np.random.normal(0.5, 1.8, size=samples)
    
    df = pd.DataFrame({
        'month': months,
        'hour': hours,
        'temperature_c': np.round(temperature, 1),
        'humidity_pct': np.round(humidity, 1),
        'pressure_hpa': np.round(pressure, 1),
        'wind_speed_kmh': np.round(wind_speed, 1),
        'cloud_cover_pct': np.round(cloud_cover, 1),
        'uv_index': uv_index,
        'weather_condition': conditions,
        'next_temp_c': np.round(next_temp, 1)
    })
    
    return df

if __name__ == '__main__':
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(data_dir, exist_ok=True)
    df = generate_synthetic_weather_data()
    file_path = os.path.join(data_dir, 'weather_dataset.csv')
    df.to_csv(file_path, index=False)
    print(f"Generated {len(df)} weather records successfully at {file_path}")
    print(df.head())
