# 🏢 Smart Building Energy Anomaly Detection
> Detecting unusual electricity consumption patterns in commercial buildings using statistical baselines, machine learning, and an LLM-powered explanation layer.

**Course:** Applied Research Methods (ARM) | M.Sc. Data Science  
**Institution:** University of Europe for Applied Sciences, Potsdam  
**Supervisor:** Shan Faiz  
**Team:** Dharmik Champaneri · Hardip Zanzmera · Sauravkumar Pandya · Dharmin Patel

---

## 📌 Project Overview

Commercial buildings account for a significant share of global energy consumption. Anomalies — unexpected spikes, drops, or sustained deviations in electricity usage — can signal equipment failure, data errors, or inefficiencies that go undetected for weeks.

This project builds an end-to-end anomaly detection pipeline for electricity meter data across four building sites (Site IDs 0–3), progressing from statistical baselines through ML models to an LLM-powered explanation layer with an interactive Streamlit dashboard.

---

## 📁 Repository Structure

```
Smart-Building-Anomaly-Detection/
│
├── data/
│   ├── processed/          # smartbuilding.db (see Dataset section below)
│   └── raw/                # Raw ASHRAE GEPIII CSVs (gitignored)
│
├── notebooks/
│   ├── sprint1_pipeline.ipynb      # ETL, EDA, data cleaning
│   ├── sprint2_features.ipynb      # Feature engineering, anomaly flagging
│   ├── sprint3_models.ipynb        # ML-based anomaly detection
│   └── sprint4_dashboard.ipynb     # LLM explanation layer + Streamlit
│
├── src/                    # Reusable helper modules
├── dashboard/              # Streamlit app (coming Sprint 4)
├── .gitignore
└── README.md
```

---

## 📦 Dataset

**Source:** [ASHRAE Great Energy Predictor III (GEPIII)](https://www.kaggle.com/c/ashrae-energy-prediction/data)  
**Scope:** Electricity meters only · Site IDs 0, 1, 2, 3

The SQLite database (`smartbuilding.db`) is too large to host on GitHub.

📥 **Download here:** [smartbuilding.db – Google Drive](https://drive.google.com/file/d/11Ko5ebeHwO-6PraaCm94gKkbNRKOsPJo/view?usp=sharing)

After downloading, place the file at:
```
data/processed/smartbuilding.db
```

---

## 🔧 Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.11 |
| Data & EDA | Pandas, NumPy, Matplotlib, Seaborn |
| Storage | SQLite via `sqlite3` |
| ML Models | scikit-learn |
| LLM Layer | Groq API |
| Dashboard | Streamlit |
| Environment | Google Colab + Google Drive |
| Version Control | GitHub |

---

## 🚀 Sprint Progress

| Sprint | Focus | Status |
|---|---|---|
| Sprint 1 | ETL pipeline, SQLite ingestion, EDA, data cleaning | ✅ Complete |
| Sprint 2 | Temporal features, rolling stats, Z-score/IQR anomaly flagging | ✅ Complete |
| Sprint 3 | ML-based anomaly detection (Isolation Forest, autoencoders) | ✅ Complete |
| Sprint 4 | Groq LLM explanation layer + Streamlit dashboard | ✅ Complete |

---

## ⚙️ Setup & Usage

### 1. Clone the repo
```bash
git clone https://github.com/DevDharmik/Smart-Building-Anomaly-Detection.git
cd Smart-Building-Anomaly-Detection
```

### 2. Download the database
Download `smartbuilding.db` from the [Google Drive link](https://drive.google.com/file/d/11Ko5ebeHwO-6PraaCm94gKkbNRKOsPJo/view?usp=sharing) and place it in `data/processed/`.

### 3. Open in Google Colab
All notebooks are designed to run in **Google Colab** with the database mounted from Google Drive. Mount your Drive at the start of each notebook:

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 4. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn streamlit groq
```

---

## 🔑 Environment Variables

Never hardcode tokens. Set the following in your Colab session:

```python
import os
os.environ['GITHUB_TOKEN'] = 'your_token_here'   # for Git pushes only
os.environ['GROQ_API_KEY'] = 'your_groq_key_here' # for LLM layer (Sprint 4)
```

---

## 📄 License

This project is developed for academic purposes as part of the M.Sc. Data Science programme at the University of Europe for Applied Sciences.
