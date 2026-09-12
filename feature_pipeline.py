import os
import sys
import time
import requests
import pandas as pd
import datetime
import hopsworks
from dotenv import load_dotenv

load_dotenv()

LAT = 33.6844
LON = 73.0479
CITY = "Islamabad"

def get_historical_data(api_key, start_date, end_date):
    """Fetches historical air pollution data from OpenWeather API"""
    start_unix = int(start_date.timestamp())
    end_unix = int(end_date.timestamp())
    
    url = f"http://api.openweathermap.org/data/2.5/air_pollution/history?lat={LAT}&lon={LON}&start={start_unix}&end={end_unix}&appid={api_key}"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"Error making OpenWeather API request: {e}")
        return pd.DataFrame()
    
    if "list" not in data or not data["list"]:
        print("Warning: No records found in OpenWeather response:", data)
        return pd.DataFrame()
        
    records = []
    for item in data["list"]:
        record = {
            "city": CITY.lower(),
            "timestamp": pd.to_datetime(item["dt"], unit="s"),
            "aqi": item["main"]["aqi"],
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
    
    # String key for Hudi record key
    df["date_str"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    
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
    df["target_aqi_next_3_days"] = df["aqi"].shift(-72)
    
    return df

def insert_with_retry(aqi_fg, df_features, max_retries=3, delay_seconds=10):
    """Inserts data into Hopsworks Feature Group with exponential retries"""
    for attempt in range(1, max_retries + 1):
        try:
            print(f"Insertion attempt {attempt}/{max_retries} to Hopsworks...")
            aqi_fg.insert(df_features, write_options={"wait_for_job": True})
            print("Successfully inserted features into Hopsworks!")
            return True
        except Exception as e:
            print(f"Insertion attempt {attempt} failed with error: {e}")
            if attempt < max_retries:
                print(f"Retrying in {delay_seconds} seconds...")
                time.sleep(delay_seconds)
                delay_seconds *= 2
            else:
                print("Max retries reached. Insertion failed.")
                raise e

def run():
    print("Starting Feature Pipeline...")
    api_key = os.getenv("OPENWEATHER_API_KEY")
    hopsworks_key = os.getenv("HOPSWORKS_API_KEY")

    if not api_key or not hopsworks_key:
        raise ValueError("Missing OPENWEATHER_API_KEY or HOPSWORKS_API_KEY")

    # Determine fetch duration: default 7 days for hourly updates, or 90 days for full backfill
    is_backfill = "--backfill" in sys.argv or os.getenv("BACKFILL", "false").lower() == "true"
    days_to_fetch = 90 if is_backfill else int(os.getenv("DAYS_TO_FETCH", "7"))
    
    project = hopsworks.login(api_key_value=hopsworks_key)
    fs = project.get_feature_store()
    
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=days_to_fetch)
    
    print(f"Fetching data for past {days_to_fetch} days ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}) for {CITY}...")
    df = get_historical_data(api_key, start_date, end_date)
    
    if df.empty:
        raise RuntimeError("No data fetched from OpenWeather API.")
        
    print(f"Fetched {len(df)} records. Computing features...")
    df_features = compute_features(df)
    
    print("Connecting to Feature Group (version 2)...")
    aqi_fg = fs.get_or_create_feature_group(
        name="islamabad_aqi_features",
        version=2,
        primary_key=["city", "date_str"],
        description="Air Quality features for Islamabad (v2)",
        event_time="timestamp"
    )
    
    insert_with_retry(aqi_fg, df_features)
    print("Feature Pipeline completed successfully!")

if __name__ == "__main__":
    run()
