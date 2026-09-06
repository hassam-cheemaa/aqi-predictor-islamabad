import os
import requests
import pandas as pd
import datetime
import hopsworks
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Coordinates for Islamabad
LAT = 33.6844
LON = 73.0479
CITY = "Islamabad"

def get_historical_data(api_key, start_date, end_date):
    """Fetches historical air pollution data from OpenWeather API"""
    start_unix = int(start_date.timestamp())
    end_unix = int(end_date.timestamp())
    
    url = f"http://api.openweathermap.org/data/2.5/air_pollution/history?lat={LAT}&lon={LON}&start={start_unix}&end={end_unix}&appid={api_key}"
    response = requests.get(url)
    data = response.json()
    
    if "list" not in data:
        print("Error fetching data:", data)
        return pd.DataFrame()
        
    records = []
    for item in data["list"]:
        record = {
            "timestamp": pd.to_datetime(item["dt"], unit="s"),
            "aqi": item["main"]["aqi"],  # AQI from 1 to 5
            "co": item["components"]["co"],
            "no": item["components"]["no"],
            "no2": item["components"]["no2"],
            "o3": item["components"]["o3"],
            "so2": item["components"]["so2"],
            "pm2_5": item["components"]["pm2_5"],
            "pm10": item["components"]["pm10"],
            "nh3": item["components"]["nh3"]
        }
        records.append(record)
        
    return pd.DataFrame(records)

def compute_features(df):
    """Computes time-based and derived features"""
    if df.empty:
        return df
        
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # Time-based features
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    
    # Derived features
    df["pm2_5_rolling_24h"] = df["pm2_5"].rolling(window=24, min_periods=1).mean()
    df["pm10_rolling_24h"] = df["pm10"].rolling(window=24, min_periods=1).mean()
    df["aqi_change_rate"] = df["aqi"].diff().fillna(0)
    
    # Target: AQI in 3 days (72 hours)
    # We create a target column by shifting the AQI backwards
    df["target_aqi_next_3_days"] = df["aqi"].shift(-72)
    
    # Drop rows where target is NaN (the last 3 days) during training backfill
    # But keep them for inference. We will handle this by returning the full df
    return df

def run():
    print("Starting Feature Pipeline...")
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        print("OPENWEATHER_API_KEY is not set. Please set it in .env or environment variables.")
        return
        
    # Connect to Hopsworks
    project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
    fs = project.get_feature_store()
    
    # For backfill, let's fetch the last 90 days
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=90)
    
    print(f"Fetching data from {start_date} to {end_date}...")
    df = get_historical_data(api_key, start_date, end_date)
    
    if df.empty:
        print("No data fetched.")
        return
        
    print("Computing features...")
    df_features = compute_features(df)
    
    # We should ensure no NaN in targets for training data, but for Hopsworks we can just upload it all
    # and filter in the training pipeline.
    
    print("Connecting to Feature Group...")
    aqi_fg = fs.get_or_create_feature_group(
        name="islamabad_aqi_features",
        version=1,
        primary_key=["timestamp"],
        description="Air Quality features for Islamabad",
        event_time="timestamp"
    )
    
    print("Inserting data to Hopsworks...")
    aqi_fg.insert(df_features, write_options={"wait_for_job" : False})
    print("Feature Pipeline completed successfully!")

if __name__ == "__main__":
    run()
