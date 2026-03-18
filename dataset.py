import numpy as np
import pandas as pd
import os

def download_real_data():
    try:
        import requests  

        print("Trying to download real Delhi weather data from Open-Meteo...")
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": 28.6139,     
            "longitude": 77.2090,      
            "start_date": "2020-01-01",
            "end_date": "2023-12-31",
            "daily": [
                "temperature_2m_max",   # daily maximum temperature
                "temperature_2m_min",   # daily minimum temperature
                "precipitation_sum",    # total rainfall that day
                "windspeed_10m_max",    # maximum wind speed
            ],
            "timezone": "Asia/Kolkata"
        }
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()

        data = response.json()

        daily = data["daily"]

        df = pd.DataFrame({
            "date":     pd.to_datetime(daily["time"]),
            "temp_max": daily["temperature_2m_max"],
            "temp_min": daily["temperature_2m_min"],
            "rainfall": daily["precipitation_sum"],
            "windspeed": daily["windspeed_10m_max"],
        })

        print(f"  Success! Downloaded {len(df)} days of real data.")
        return df

    except Exception as e:
        print(f"  Could not download real data: {e}")
        print("  Switching to generated data instead.")
        return None

def generate_delhi_data():
    print("Generating realistic Delhi weather data (offline mode)...")

    np.random.seed(42)

    dates = pd.date_range(start="2020-01-01", end="2023-12-31", freq="D")
    n = len(dates) 

    temp_max = []
    temp_min = []
    rainfall  = []
    windspeed = []

    for date in dates:
        doy = date.day_of_year  
        month = date.month

        seasonal_base = 29.5 - 12.5 * np.cos(2 * np.pi * (doy - 15) / 365)

        noise = np.random.normal(0, 2.5)
        tmax = seasonal_base + noise

        tmin = tmax - np.random.uniform(7, 12)

        if month in [6, 7, 8, 9]:

            if np.random.random() < 0.60:
                rain = np.random.exponential(12) 
            else:
                rain = 0.0
        elif month in [12, 1, 2]:
            if np.random.random() < 0.10:
                rain = np.random.uniform(1, 8)
            else:
                rain = 0.0
        else:
            if np.random.random() < 0.05:
                rain = np.random.uniform(0.5, 5)
            else:
                rain = 0.0

        if month in [6, 7, 8, 9] and rain > 5:
            tmax -= np.random.uniform(1, 4)

        if month in [4, 5, 6]:
            wind = np.random.uniform(15, 35)  
        else:
            wind = np.random.uniform(5, 20)   

        temp_max.append(round(tmax, 1))
        temp_min.append(round(tmin, 1))
        rainfall.append(round(max(rain, 0), 1))  
        windspeed.append(round(wind, 1))

    df = pd.DataFrame({
        "date":      dates,
        "temp_max":  temp_max,
        "temp_min":  temp_min,
        "rainfall":  rainfall,
        "windspeed": windspeed,
    })

    print(f"  Generated {len(df)} days of synthetic Delhi data.")
    return df


def engineer_features(df):
    print("Engineering features...")

    df = df.copy()

    df = df.sort_values("date").reset_index(drop=True)

    df["temp_max_lag1"] = df["temp_max"].shift(1)   
    df["temp_max_lag2"] = df["temp_max"].shift(2)   
    df["temp_max_lag3"] = df["temp_max"].shift(3)   
    df["temp_min_lag1"] = df["temp_min"].shift(1)   

    df["temp_rolling_7d"] = df["temp_max"].shift(1).rolling(window=7, min_periods=7).mean()

    df["temp_range_lag1"] = df["temp_max_lag1"] - df["temp_min_lag1"]

    df["rainfall_lag1"] = df["rainfall"].shift(1)

    df["month"] = df["date"].dt.month

    df["day_of_year"] = df["date"].dt.day_of_year

    df = df.dropna().reset_index(drop=True)

    print(f"  Features engineered. Dataset has {len(df)} rows and {len(df.columns)} columns.")
    return df

def get_dataset(save_path="delhi_weather.csv"):

    df_raw = download_real_data()

    if df_raw is None:
        df_raw = generate_delhi_data()

    df = engineer_features(df_raw)

    df.to_csv(save_path, index=False)
    print(f"\nDataset saved to '{save_path}'")
    print(f"Total rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst 3 rows:")
    print(df.head(3).to_string())

    return df

if __name__ == "__main__":
    get_dataset()