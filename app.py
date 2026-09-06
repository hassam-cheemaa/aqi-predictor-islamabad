import streamlit as st
import hopsworks
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Islamabad AQI Predictor", layout="wide")

@st.cache_resource
def get_hopsworks_project():
    api_key = os.getenv("HOPSWORKS_API_KEY")
    if not api_key:
        st.error("HOPSWORKS_API_KEY not set. Add it in Streamlit Secrets or your .env file.")
        st.stop()
    return hopsworks.login(api_key_value=api_key)

def get_model_and_features(project):
    # Fetch latest model dynamically from registry
    mr = project.get_model_registry()
    models = mr.get_models("islamabad_aqi_model")
    if not models:
        raise ValueError("No trained model found in Hopsworks Model Registry. Please run training_pipeline.py first.")
    model_obj = models[-1]
    model_dir = model_obj.download()
    model = joblib.load(f"{model_dir}/aqi_model.pkl")
    
    # Fetch latest features
    fs = project.get_feature_store()
    try:
        feature_view = fs.get_feature_view(name="islamabad_aqi_fv", version=2)
    except Exception:
        feature_view = None
        
    if feature_view is None:
        aqi_fg = fs.get_feature_group(name="islamabad_aqi_features", version=2)
        feature_view = fs.create_feature_view(
            name="islamabad_aqi_fv",
            version=2,
            description="Feature view for AQI prediction (v2)",
            query=aqi_fg.select_all()
        )
        
    df = feature_view.get_batch_data()
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp", ascending=False).reset_index(drop=True)
    return model, df

st.title("Air Quality Index (AQI) Predictor - Islamabad")
st.write("Predicting the AQI in Islamabad for the next 3 days using a 100% serverless ML pipeline.")

try:
    project = get_hopsworks_project()
    model, df_features = get_model_and_features(project)
    
    latest_data = df_features.iloc[0:1]
    timestamp = latest_data["timestamp"].values[0]
    current_aqi = latest_data["aqi"].values[0]
    
    st.header(f"Current AQI (as of {pd.to_datetime(timestamp).strftime('%Y-%m-%d %H:%M')})")
    
    # Display Current AQI Alert
    # OpenWeather AQI scale: 1 = Good, 2 = Fair, 3 = Moderate, 4 = Poor, 5 = Very Poor
    if current_aqi <= 1:
        st.success(f"Current AQI: {current_aqi} (Good)")
    elif current_aqi <= 2:
        st.info(f"Current AQI: {current_aqi} (Fair)")
    elif current_aqi <= 3:
        st.warning(f"Current AQI: {current_aqi} (Moderate)")
    elif current_aqi <= 4:
        st.error(f"Current AQI: {current_aqi} (Poor)")
    else:
        st.error(f"⚠️ HAZARDOUS Current AQI: {current_aqi} (Very Poor) ⚠️")
        
    # Prepare features for prediction using exact feature names from training
    if hasattr(model, "feature_names_in_"):
        X_pred = latest_data[model.feature_names_in_]
    else:
        X_pred = latest_data.drop(columns=["target_aqi_next_3_days", "timestamp", "city", "date_str"], errors='ignore')
    
    predicted_aqi = float(model.predict(X_pred)[0])
    
    st.header("Forecast: AQI in 3 Days")
    st.metric(
        label="Predicted AQI (72 hours)", 
        value=f"{predicted_aqi:.2f}", 
        delta=f"{predicted_aqi - current_aqi:+.2f} from now", 
        delta_color="inverse"
    )
    
    if predicted_aqi >= 4:
        st.error("⚠️ ALERT: The predicted AQI is expected to reach unhealthy/hazardous levels in 3 days. Please take precautions.")
    elif predicted_aqi <= 2:
        st.success("Air quality is projected to remain acceptable over the next 3 days.")
    else:
        st.warning("Moderate air quality expected over the next 3 days.")
        
    # SHAP Explanations
    st.header("Model Explanation (SHAP)")
    st.write("What features are driving this prediction?")
    
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_pred)
        fig, ax = plt.subplots(figsize=(10, 5))
        shap.plots.waterfall(shap_values[0], show=False)
        plt.tight_layout()
        st.pyplot(fig)
    except Exception:
        # Fallback for alternative shap versions
        try:
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(X_pred)
            fig, ax = plt.subplots(figsize=(10, 5))
            base_val = explainer.expected_value[0] if isinstance(explainer.expected_value, (list, tuple, np.ndarray)) else explainer.expected_value
            shap.waterfall_plot(shap.Explanation(values=sv[0], base_values=base_val, data=X_pred.iloc[0], feature_names=X_pred.columns))
            plt.tight_layout()
            st.pyplot(fig)
        except Exception as e_shap:
            st.info("Feature Importance Breakdown:")
            importances = pd.Series(model.feature_importances_, index=X_pred.columns).sort_values(ascending=False)
            st.bar_chart(importances.head(8))
    
    st.header("Recent Historical Pollution Trend")
    st.line_chart(df_features.head(72).set_index("timestamp")[["pm2_5", "pm10", "no2", "o3"]])
    
except Exception as e:
    st.error(f"Failed to load data or model: {e}")
    st.write("Make sure you have run feature_pipeline.py and training_pipeline.py so data & models are available in Hopsworks.")
