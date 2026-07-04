# Smart Building Anomaly Detection

Multi-dataset anomaly detection system for smart building IoT sensor data, combining statistical and machine learning methods with LLM-generated natural language explanations.

Built as part of the M.Sc. Data Science program at the **University of Europe for Applied Sciences (UEAS), Potsdam**.

**Live Dashboard:** [Streamlit App](https://smart-building-anomaly-detection.streamlit.app) *(update with your actual deployed URL)*
**Repository:** `github.com/DevDharmik/Smart-Building-Anomaly-Detection`

---

## Overview

Buildings generate continuous streams of energy, temperature, humidity, and occupancy data. Detecting anomalies in these streams — equipment faults, unusual consumption spikes, sensor drift — is critical for operational efficiency and predictive maintenance.

This project implements a full anomaly detection pipeline across **two independent smart building datasets**, letting us validate detection methods across different building types, climates, and sensor configurations:

| Dataset | Description | Granularity |
|---|---|---|
| **ASHRAE GEPIII** | Global Energy Prediction competition dataset — multi-building electricity meter readings | Hourly, per building |
| **Sharjah 2024 IoT** | University of Sharjah smart building dataset (Jan–Jun 2024) — power, temperature, humidity, motion | High-frequency, per appliance/room |

## Methodology

Three complementary anomaly detection methods are applied to every stream, with a consensus flag when 2 of 3 agree:

- **Z-score** — flags points beyond 3 standard deviations from the group mean
- **IQR (Interquartile Range)** — flags points outside `Q1 - 1.5×IQR` / `Q3 + 1.5×IQR`
- **Isolation Forest** — unsupervised ML model isolating outliers via random partitioning

Flagged anomalies are optionally explained in natural language using **Groq's `llama-3.3-70b-versatile`**, which takes the anomaly's statistical context (value, rolling mean/std, time features) and generates a plausible cause + recommended action for a building manager.

## Tech Stack

- **Data processing:** Python, Pandas, NumPy
- **ML:** Scikit-learn (Isolation Forest, MinMaxScaler)
- **Storage:** SQLite
- **Visualization:** Matplotlib, Seaborn
- **Dashboard:** Streamlit
- **LLM explanations:** Groq API (`llama-3.3-70b-versatile`)
- **Environment:** Google Colab, Google Drive

## Repository Structure

```
├── notebooks/
│   ├── 01-04_gepiii_*.ipynb          # ASHRAE GEPIII pipeline (4 sprints)
│   ├── 05_sharjah_2024_ingestion.ipynb
│   └── 06_sharjah_2024_anomaly_detection.ipynb
├── app.py                             # Streamlit dashboard (dataset toggle: GEPIII / Sharjah)
├── outputs/
│   └── plots/                         # Generated visualizations
└── README.md
```

## Dashboard Features

- Toggle between ASHRAE GEPIII and Sharjah 2024 datasets
- Per-building / per-appliance / per-location drill-down
- Switchable detection method (Z-score, IQR, Isolation Forest, Consensus)
- Hourly and day-of-week anomaly rate breakdowns
- Live LLM-generated anomaly explanations via Groq

## Running Locally

```bash
git clone https://github.com/DevDharmik/Smart-Building-Anomaly-Detection.git
cd Smart-Building-Anomaly-Detection
pip install -r requirements.txt
streamlit run app.py
```

You'll need a free Groq API key from [console.groq.com](https://console.groq.com) to use the live explanation feature.

## Team

- Dharmik Champaneri
- Hardip Zanzmera
- Sauravkumar Pandya
- Dharmin Patel

Supervised by Shan Faiz — UEAS Potsdam.

## License

MIT
