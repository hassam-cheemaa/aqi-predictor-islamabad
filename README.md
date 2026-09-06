# 🍃 Islamabad Air Quality Index (AQI) Predictor

[![Live App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://aqi-predictor-isl.streamlit.app)
[![Feature Pipeline](https://github.com/hassam-cheemaa/aqi-predictor-islamabad/actions/workflows/feature_pipeline.yml/badge.svg)](https://github.com/hassam-cheemaa/aqi-predictor-islamabad/actions/workflows/feature_pipeline.yml)
[![Training Pipeline](https://github.com/hassam-cheemaa/aqi-predictor-islamabad/actions/workflows/training_pipeline.yml/badge.svg)](https://github.com/hassam-cheemaa/aqi-predictor-islamabad/actions/workflows/training_pipeline.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Hopsworks](https://img.shields.io/badge/Hopsworks-Feature%20Store%20%26%20Model%20Registry-brightgreen)](https://www.hopsworks.ai/)

> **Live Dashboard**: [https://aqi-predictor-isl.streamlit.app](https://aqi-predictor-isl.streamlit.app)

An end-to-end, **100% serverless Machine Learning system** that monitors real-time air pollution and forecasts the Air Quality Index (AQI) in **Islamabad, Pakistan** for the next **3 days (72 hours)**.

Built with automated hourly data ingestion, daily model retraining, feature store versioning, explainable AI (SHAP), and a real-time Streamlit dashboard providing actionable health advisories.

---

## 📌 Project Overview

Urban centers across South Asia face acute seasonal air pollution challenges. Islamabad, nestled against the Margalla Hills, frequently experiences trapping of fine particulate matter ($PM_{2.5}$ and $PM_{10}$), leading to rapid shifts between moderate and hazardous air quality.

Traditional air quality portals only tell you what the air was like *in the past*. This project bridges that gap by building a self-sustaining serverless ML system that:
1. **Continuously collects** real-time pollutant measurements ($PM_{2.5}$, $PM_{10}$, $NO_2$, $O_3$, $SO_2$, $CO$) every hour.
2. **Engineers time-series & rolling features** and stores them in a cloud-hosted feature store.
3. **Trains and updates** predictive models daily to forecast AQI levels 72 hours into the future.
4. **Interprets predictions** using SHAP (Shapley Additive Explanations) so users understand *why* the air quality is expected to change.
5. **Provides dynamic health advisories** (mask recommendations, exercise safety, and sensitive group precautions) based on current air quality.

---

## 🏛️ System Architecture

```
                       ┌───────────────────────────────┐
                       │  OpenWeather Air Quality API  │
                       └───────────────┬───────────────┘
                                       │ Raw hourly data
                                       ▼
                       ┌───────────────────────────────┐
                       │  Hourly Feature Pipeline      │
                       │  (GitHub Actions CI/CD)       │
                       └───────────────┬───────────────┘
                                       │ Materialized features
                                       ▼
                       ┌───────────────────────────────┐
                       │   Hopsworks Feature Store     │
                       │   Feature Group: v2 (Hudi)    │
                       └───────────────┬───────────────┘
                                       │
                    ┌──────────────────┴──────────────────┐
                    │                                     │
                    ▼                                     ▼
     ┌───────────────────────────────┐     ┌───────────────────────────────┐
     │  Daily Training Pipeline      │     │  Interactive Web Dashboard    │
     │  (GitHub Actions CI/CD)       │     │  (Streamlit Community Cloud)  │
     └──────────────┬────────────────┘     └──────────────┬────────────────┘
                    │                                     │
                    ▼                                     │
     ┌───────────────────────────────┐                    │
     │   Hopsworks Model Registry    │◄───────────────────┘
     │   islamabad_aqi_model (v1)    │  Loads latest model
     └───────────────────────────────┘  & real-time features
```

---

## 📊 Model Performance & Evaluation Metrics

The forecasting engine utilizes an ensemble **Random Forest Regressor** trained on 90 days of hourly historical air quality records (2,016 validated time-series samples).

### Evaluation Results on Hold-out Test Set:

| Metric | Score | Interpretation |
| :--- | :---: | :--- |
| **$R^2$ Score (Coefficient of Determination)** | **`0.8106`** | Explains **81.1%** of variance in future 72-hour air quality levels. |
| **Root Mean Squared Error (RMSE)** | **`0.3280`** | Low penalty for large deviations across the discrete index scale. |
| **Mean Absolute Error (MAE)** | **`0.2162`** | Average forecast deviation is less than a quarter of an AQI category point. |

All trained models and their schemas, hyperparameters, and evaluation metrics are systematically versioned in the **Hopsworks Model Registry**.

---

## 🛠️ Technology Stack & Tools

| Category | Tool / Library | Purpose in this Project |
| :--- | :--- | :--- |
| **Language & Runtime** | Python 3.11 | Primary programming environment across pipelines and application. |
| **Feature Store & Registry** | [Hopsworks](https://www.hopsworks.ai/) | Centralized feature storage, feature view generation, and model registry. |
| **Machine Learning** | Scikit-learn (`RandomForestRegressor`) | Non-linear regression modeling for 72-hour AQI prediction. |
| **Explainable AI** | [SHAP](https://shap.readthedocs.io/) | TreeExplainer waterfall plots identifying top pollutant contributors. |
| **CI/CD Automation** | GitHub Actions | Scheduled hourly feature collection and daily automated model retraining. |
| **Web Dashboard** | Streamlit | Responsive, dark-themed user interface with real-time health indicators. |
| **Data Ingestion** | OpenWeather API | Hourly atmospheric pollution data ($PM_{2.5}, PM_{10}, NO_2, O_3, SO_2, CO$). |
| **Cross-Validation** | Open-Meteo & IQAir | Multi-source ground and satellite validation benchmarks. |
| **Data Processing** | Pandas, NumPy | Time-series rolling windows, lag calculations, and target generation. |

---

## 🧪 Feature Engineering

From raw atmospheric readings, the pipeline generates:
* **Temporal Features**: `hour`, `day`, `month`, `dayofweek` (captures diurnal traffic cycles and seasonal shifts).
* **Rolling Trend Features**:
  * `pm2_5_rolling_24h`: 24-hour moving average of fine particulate matter.
  * `pm10_rolling_24h`: 24-hour moving average of coarse particulate matter.
* **Dynamic Indicators**: `aqi_change_rate` (rate of change between consecutive hours).
* **Supervised Target**: `target_aqi_next_3_days` (future AQI shifted backwards by 72 hours for predictive alignment).

---

## 📈 Understanding AQI: Dual-Scale Display

Air quality reporting varies internationally. To prevent confusion, the live dashboard presents both standards:

1. **US EPA AQI Scale (0 to 500)**: The standard recognized by Pakistani media and IQAir. It is calculated dynamically from real-time $PM_{2.5}$ concentrations using the official EPA piecewise formula:
   $$\text{AQI} = \frac{I_{high} - I_{low}}{C_{high} - C_{low}} \times (C - C_{low}) + I_{low}$$
2. **OpenWeather / European CAQI Scale (1 to 5)**:
   * **Level 1**: Good (0–50 US AQI equivalent)
   * **Level 2**: Fair (51–100 US AQI equivalent)
   * **Level 3**: Moderate (101–150 US AQI equivalent)
   * **Level 4**: Poor (151–200 US AQI equivalent)
   * **Level 5**: Very Poor (201+ US AQI equivalent)

---

## 🛡️ Dynamic Health Advisories

The dashboard automatically updates health recommendations tailored to current pollution severity:

* **Good (0–50)**: Normal outdoor recreation; ideal time for outdoor exercise.
* **Moderate (51–100)**: Acceptable air; sensitive individuals should reduce strenuous outdoor exertion.
* **Unhealthy for Sensitive Groups (101–150)**: Children, elderly, and respiratory patients are advised to reduce outdoor exertion and keep masks handy.
* **Unhealthy / Poor (151–200)**: Everyone should avoid prolonged outdoor workouts; N95 masks recommended; keep windows closed.
* **Hazardous (201+)**: Emergency health alert; run indoor air purifiers; avoid going outside unless essential.

---

## 🚀 Local Development Setup

### 1. Clone the repository
```bash
git clone https://github.com/hassam-cheemaa/aqi-predictor-islamabad.git
cd aqi-predictor-islamabad
```

### 2. Create virtual environment & install dependencies
```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Set environment variables
Create a `.env` file in the root directory:
```env
HOPSWORKS_API_KEY="your_hopsworks_api_key"
OPENWEATHER_API_KEY="your_openweather_api_key"
```

### 4. Run pipelines locally
```bash
# 1. Fetch data & insert into Hopsworks
python feature_pipeline.py

# 2. Train model & register into Hopsworks Model Registry
python training_pipeline.py

# 3. Launch interactive Streamlit dashboard
streamlit run app.py
```

---

## 🔄 Automated CI/CD Workflows

The serverless architecture runs autonomously without dedicated server maintenance:

* **Hourly Feature Pipeline (`.github/workflows/feature_pipeline.yml`)**:
  * Runs every hour (`0 * * * *`).
  * Fetches latest Islamabad air quality data, computes features, and commits records to Hopsworks.
* **Daily Training Pipeline (`.github/workflows/training_pipeline.yml`)**:
  * Runs daily at midnight (`0 0 * * *`).
  * Ingests latest feature batches, retrains the Random Forest regressor, evaluates metrics, and logs the newest model version.

---

## 🌐 Live Application

The web interface is hosted publicly on Streamlit Community Cloud:
👉 **[https://aqi-predictor-isl.streamlit.app](https://aqi-predictor-isl.streamlit.app)**

---

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).
