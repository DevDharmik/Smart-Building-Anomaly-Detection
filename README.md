# 🏢 Smart Building Energy Anomaly Detection

> Detecting unusual electricity consumption patterns in commercial buildings using statistical and machine learning methods on the ASHRAE Great Energy Predictor III dataset.

**Course:** Applied Research Methods (ARM) | M.Sc. Data Science  
**Institution:** University of Europe for Applied Sciences, Potsdam  
**Supervisor:** Shan Faiz  
**Team:** Dharmik Champaneri · Hardip Zanzmera · Sauravkumar Pandya · Dharmin Patel

---

## 📌 Project Overview

Commercial buildings account for a significant share of global energy consumption. Anomalies — unexpected spikes, drops, or sustained deviations in electricity usage — can signal equipment failure, data errors, or inefficiencies that go undetected for weeks.

This project builds an end-to-end anomaly detection pipeline for electricity meter data across four building sites, progressing from statistical baselines through to ML models and an LLM-powered explanation layer.

---

## 📁 Repository Structure

Smart-Building-Anomaly-Detection/

│

├── notebooks/

│   ├── 01_eda.ipynb                  # Sprint 1: Data pipeline + EDA

│   └── 02_features_baselines.ipynb   # Sprint 2: Feature engineering + Z-score/IQR

│

├── data/

│   └── processed/

│       ├── distribution_plot.png

│       ├── timeseries_sample.png

│       ├── hourly_pattern.png

│       ├── zscore_anomalies.png

│       ├── iqr_anomalies.png

│       ├── anomaly_by_hour.png

│       └── anomaly_by_day.png

│

├── .gitignore

├── LICENSE

└── README.md

---

## 📊 Dataset

**ASHRAE Great Energy Predictor III** (Kaggle)

| File | Description |
|------|-------------|
| `train.csv` | Hourly meter readings per building |
| `building_metadata.csv` | Site, primary use, floor area, year built |
| `weather_train.csv` | Hourly weather data per site |

**Subset used:** Electricity meters (`meter == 0`), Site IDs 0–3  
**Scope:** ~1 year of hourly readings across multiple commercial buildings

---

## 🔬 Sprint Progress

### ✅ Sprint 1 — Data Pipeline + EDA (`01_eda.ipynb`)

- Loaded and merged train + building metadata
- Filtered to electricity meters, sites 0–3
- Removed negative readings; interpolated gaps ≤ 3 hours
- Stored cleaned data in SQLite (`smart_building.db`)
- EDA: distribution plots, time series samples, hourly consumption patterns, site-level comparison
- Estimated baseline anomaly rate using 99th/1st percentile thresholds

### ✅ Sprint 2 — Feature Engineering + Statistical Baselines (`02_features_baselines.ipynb`)

**Features engineered:**
- Temporal: `hour`, `day_of_week`, `month`, `is_weekend`, `day_of_year`
- Rolling 24hr `mean` and `std` per building
- 7-day lag feature (same hour, 168 readings prior)

**Anomaly detection baselines:**
- **Z-score flagging** — flags readings where |z| > 3 relative to 24hr rolling mean
- **IQR flagging** — flags readings outside 1.5× IQR band of rolling 24hr window
- **Combined label** — union of Z-score OR IQR flags

All features saved to SQLite (`energy_features` table).

### 🔜 Sprint 3 — Isolation Forest + LSTM Autoencoder + SHAP
### 🔜 Sprint 4 — LLM Explanation Layer (Groq API) + Streamlit Dashboard

---

## 🛠️ Tech Stack

| Layer | Tools |
|-------|-------|
| Language | Python 3.10 |
| Data | Pandas, NumPy, SQLite |
| Visualisation | Matplotlib, Seaborn, Plotly |
| ML (planned) | scikit-learn, TensorFlow/Keras |
| Explainability (planned) | SHAP |
| LLM Layer (planned) | Groq API |
| Dashboard (planned) | Streamlit |
| Environment | Google Colab + Google Drive |

---

## ⚙️ Setup & Usage

All notebooks are designed to run on **Google Colab**.

1. Upload `ashrae-energy-prediction.zip` to your Google Drive root
2. Clone the repo in Colab:
```python
!git clone https://github.com/DevDharmik/Smart-Building-Anomaly-Detection.git
```
3. Mount Drive and run notebooks in order:
   - `01_eda.ipynb` → `02_features_baselines.ipynb`

> **Note:** Never hardcode GitHub tokens. Use Colab Secrets (`Tools → Secrets → GITHUB_TOKEN`).

---

## 📈 Key Results (Sprints 1–2)

| Metric | Value |
|--------|-------|
| Filtered dataset size | ~1.5M rows |
| Buildings in scope | 100+ |
| Z-score anomaly rate | ~2% |
| IQR anomaly rate | ~8% |
| Combined label rate | ~9% |

---

## 👥 Team

| Name | GitHub |
|------|--------|
| Dharmik Champaneri | [@DevDharmik](https://github.com/DevDharmik) |
| Hardip Zanzmera | — |
| Sauravkumar Pandya | — |
| Dharmin Patel | — |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
