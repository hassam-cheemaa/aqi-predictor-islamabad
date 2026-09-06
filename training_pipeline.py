import os
import hopsworks
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import joblib
from hsml.schema import Schema
from hsml.model_schema import ModelSchema
from dotenv import load_dotenv

load_dotenv()

def run():
    print("Starting Training Pipeline...")
    hopsworks_key = os.getenv("HOPSWORKS_API_KEY")
    if not hopsworks_key:
        raise ValueError("HOPSWORKS_API_KEY is not set.")

    project = hopsworks.login(api_key_value=hopsworks_key)
    fs = project.get_feature_store()
    
    # Retrieve feature group version 2
    aqi_fg = fs.get_feature_group(name="islamabad_aqi_features", version=2)
    query = aqi_fg.select_all()
    
    try:
        feature_view = fs.get_feature_view(name="islamabad_aqi_fv", version=2)
    except:
        feature_view = fs.create_feature_view(
            name="islamabad_aqi_fv",
            version=2,
            description="Feature view for AQI prediction (v2)",
            query=query
        )
    
    print("Fetching training data from Hopsworks Feature Store...")
    df = feature_view.get_batch_data()
    
    # Deduplicate and drop rows missing target
    df = df.drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
    df = df.dropna(subset=["target_aqi_next_3_days"])
    
    print(f"Training on {len(df)} historical samples...")
    
    # Features and Target
    X = df.drop(columns=["target_aqi_next_3_days", "timestamp", "city", "date_str"], errors="ignore")
    y = df["target_aqi_next_3_days"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training Random Forest Regressor...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae = float(mean_absolute_error(y_test, preds))
    r2 = float(r2_score(y_test, preds))
    
    print(f"Model Evaluation Metrics -> RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
    
    model_dir = "aqi_model"
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    joblib.dump(model, f"{model_dir}/aqi_model.pkl")
    
    print("Registering model into Hopsworks Model Registry...")
    mr = project.get_model_registry()
    
    input_schema = Schema(X_train)
    output_schema = Schema(y_train)
    model_schema = ModelSchema(input_schema, output_schema)
    
    aqi_model = mr.python.create_model(
        name="islamabad_aqi_model", 
        metrics={"RMSE": rmse, "MAE": mae, "R2": r2},
        model_schema=model_schema,
        description="Random Forest model for AQI prediction"
    )
    
    aqi_model.save(model_dir)
    print("Training Pipeline completed successfully!")

if __name__ == "__main__":
    run()
