# ============================================================
# dataset.py
# ------------------------------------------------------------
# This file does ONE job: get Delhi weather data and save it
# as a CSV file called "delhi_weather.csv"
#
# It first tries to download REAL historical data from the
# Open-Meteo API (free, no account needed).
# If there is no internet, it generates realistic fake data
# based on Delhi's actual seasonal temperature patterns.
# ============================================================

import numpy as np
import pandas as pd
import os

# ----------------------------------------------------------
# PART 1: Try to download real data from Open-Meteo API
# ----------------------------------------------------------
# Open-Meteo is a free weather API. We ask it for Delhi's
# daily max/min temperature from Jan 2020 to Dec 2023.
# Delhi coordinates: latitude=28.6, longitude=77.2

def download_real_data():
    """
    Downloads real Delhi weather data from Open-Meteo.
    Returns a DataFrame if successful, None if it fails.
    """
    try:
        import requests  # requests is Python's library for making web calls

        print("Trying to download real Delhi weather data from Open-Meteo...")

        # This is the API URL. Each parameter tells the server what we want.
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": 28.6139,       # Delhi latitude
            "longitude": 77.2090,      # Delhi longitude
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

        # Make the actual web request (like opening a URL in your browser)
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()  # raises an error if request failed

        # The API returns JSON data. We convert it to a Python dictionary.
        data = response.json()

        # Extract the 'daily' section which has our actual numbers
        daily = data["daily"]

        # Build a pandas DataFrame (like an Excel table) from the response
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
        # If anything goes wrong (no internet, API down, etc.), we catch
        # the error here and return None so the program can use fake data.
        print(f"  Could not download real data: {e}")
        print("  Switching to generated data instead.")
        return None


# ----------------------------------------------------------
# PART 2: Generate realistic fake Delhi data (offline mode)
# ----------------------------------------------------------
# Delhi has very clear seasons:
#   Jan–Feb : Cold winter   (~15-20°C max)
#   Mar–May : Hot summer    (~35-45°C max)
#   Jun–Sep : Monsoon       (~33-38°C max, lots of rain)
#   Oct–Dec : Pleasant cool (~25-30°C max)
#
# We use this knowledge to generate data that LOOKS real.

def generate_delhi_data():
    """
    Creates 4 years of realistic synthetic Delhi weather.
    Returns a DataFrame with the same columns as real data.
    """
    print("Generating realistic Delhi weather data (offline mode)...")

    # np.random.seed makes sure we get the same "random" numbers every time
    # This means your results will be reproducible (same output every run)
    np.random.seed(42)

    # Create a list of every date from Jan 1 2020 to Dec 31 2023
    dates = pd.date_range(start="2020-01-01", end="2023-12-31", freq="D")
    n = len(dates)  # total number of days (should be ~1461)

    # We'll build each column separately then combine them

    # --- Temperature Max ---
    # Delhi's average max temperature follows a roughly sinusoidal (wave) pattern
    # through the year. We calculate the "day of year" (1-365) and use a
    # cosine wave to model the seasonal cycle.
    # The cosine wave peaks in summer (around day 150, which is ~June)
    # and dips in winter (around day 0/365, which is ~January)

    temp_max = []
    temp_min = []
    rainfall  = []
    windspeed = []

    for date in dates:
        doy = date.day_of_year  # day of year: 1 (Jan 1) to 365 (Dec 31)
        month = date.month

        # Seasonal base temperature using a cosine wave
        # -cos shifts the peak to summer (June/July)
        # The wave goes from roughly 17°C (winter) to 42°C (summer peak)
        seasonal_base = 29.5 - 12.5 * np.cos(2 * np.pi * (doy - 15) / 365)

        # Add random day-to-day variation (weather is never perfectly smooth)
        # np.random.normal(0, 2) means: random number centred at 0, spread of 2
        noise = np.random.normal(0, 2.5)
        tmax = seasonal_base + noise

        # Min temperature is usually 7-12 degrees below max
        tmin = tmax - np.random.uniform(7, 12)

        # --- Rainfall ---
        # Delhi gets almost all its rain during monsoon (June-September)
        # Outside monsoon, rain is rare
        if month in [6, 7, 8, 9]:
            # During monsoon: 60% chance of some rain on any given day
            if np.random.random() < 0.60:
                rain = np.random.exponential(12)  # exponential gives realistic rain amounts
            else:
                rain = 0.0
        elif month in [12, 1, 2]:
            # Winter: occasional light rain (western disturbances)
            if np.random.random() < 0.10:
                rain = np.random.uniform(1, 8)
            else:
                rain = 0.0
        else:
            # Other months: very rare rain
            if np.random.random() < 0.05:
                rain = np.random.uniform(0.5, 5)
            else:
                rain = 0.0

        # Monsoon days are slightly cooler due to cloud cover and evaporation
        if month in [6, 7, 8, 9] and rain > 5:
            tmax -= np.random.uniform(1, 4)

        # --- Wind Speed ---
        # Pre-monsoon (April-May) has strong hot winds ("loo" winds)
        if month in [4, 5, 6]:
            wind = np.random.uniform(15, 35)  # strong winds km/h
        else:
            wind = np.random.uniform(5, 20)   # normal winds

        temp_max.append(round(tmax, 1))
        temp_min.append(round(tmin, 1))
        rainfall.append(round(max(rain, 0), 1))  # rainfall can't be negative
        windspeed.append(round(wind, 1))

    # Combine everything into a DataFrame
    df = pd.DataFrame({
        "date":      dates,
        "temp_max":  temp_max,
        "temp_min":  temp_min,
        "rainfall":  rainfall,
        "windspeed": windspeed,
    })

    print(f"  Generated {len(df)} days of synthetic Delhi data.")
    return df


# ----------------------------------------------------------
# PART 3: Feature Engineering
# ----------------------------------------------------------
# Raw data alone is not enough for our model.
# We need to CREATE new columns (features) that help the model
# understand patterns. For example:
#   - What was yesterday's temperature? (lag feature)
#   - What's the average temperature over the last week? (rolling feature)
#   - What month is it? (seasonal feature)
#
# This is called "Feature Engineering" — one of the most
# important skills in real-world data science.

def engineer_features(df):
    """
    Takes the raw weather DataFrame and adds useful columns
    that will help the model make better predictions.
    """
    print("Engineering features...")

    # Make a copy so we don't accidentally modify the original
    df = df.copy()

    # Sort by date (just to be safe — data should already be sorted)
    df = df.sort_values("date").reset_index(drop=True)

    # --- Lag Features ---
    # "lag" means "look back in time"
    # lag_1 = yesterday's value, lag_2 = two days ago, etc.
    # These are powerful because tomorrow's temperature is
    # strongly correlated with today's (weather has inertia)

    df["temp_max_lag1"] = df["temp_max"].shift(1)   # yesterday's max temp
    df["temp_max_lag2"] = df["temp_max"].shift(2)   # two days ago's max temp
    df["temp_max_lag3"] = df["temp_max"].shift(3)   # three days ago
    df["temp_min_lag1"] = df["temp_min"].shift(1)   # yesterday's min temp

    # --- Rolling Mean ---
    # The average of the last 7 days captures the "trend" —
    # are we in a warming spell or cooling spell?
    # min_periods=7 means we only compute it when we have at least 7 days of history
    df["temp_rolling_7d"] = df["temp_max"].shift(1).rolling(window=7, min_periods=7).mean()

    # --- Temperature Range (yesterday) ---
    # Large range = clear sunny day. Small range = cloudy/rainy day.
    # This feature captures weather "type"
    df["temp_range_lag1"] = df["temp_max_lag1"] - df["temp_min_lag1"]

    # --- Rainfall yesterday ---
    # If it rained yesterday, today might be cooler
    df["rainfall_lag1"] = df["rainfall"].shift(1)

    # --- Month ---
    # Month captures seasonal patterns. January is always different from July.
    # We store it as a number (1=January, 12=December)
    df["month"] = df["date"].dt.month

    # --- Day of Year ---
    # More precise than month — captures gradual seasonal changes
    df["day_of_year"] = df["date"].dt.day_of_year

    # --- Remove rows where lag features are missing ---
    # The first few rows won't have lag values (there's no "yesterday" for day 1)
    # We drop these rows using dropna() which removes any row with a NaN (missing value)
    df = df.dropna().reset_index(drop=True)

    print(f"  Features engineered. Dataset has {len(df)} rows and {len(df.columns)} columns.")
    return df


# ----------------------------------------------------------
# PART 4: Main execution — run when this file is called
# ----------------------------------------------------------

def get_dataset(save_path="delhi_weather.csv"):
    """
    The main function of this file.
    Gets data (real or synthetic), engineers features,
    saves to CSV, and returns the final DataFrame.
    """

    # Step 1: Get raw data
    df_raw = download_real_data()

    # If download failed, use generated data
    if df_raw is None:
        df_raw = generate_delhi_data()

    # Step 2: Add engineered features
    df = engineer_features(df_raw)

    # Step 3: Save to CSV so we don't need to regenerate every time
    df.to_csv(save_path, index=False)
    print(f"\nDataset saved to '{save_path}'")
    print(f"Total rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst 3 rows:")
    print(df.head(3).to_string())

    return df


# This block runs ONLY when you execute "python dataset.py" directly
# It does NOT run when another file imports from dataset.py
if __name__ == "__main__":
    get_dataset()