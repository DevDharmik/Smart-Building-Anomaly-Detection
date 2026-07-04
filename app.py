import os
import streamlit as st
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
from groq import Groq
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="Smart Building Anomaly Detection",
                   page_icon="🏢", layout="wide")

GEPIII_DB_PATH   = "smart_building.db"
GEPIII_GDRIVE_ID = "11Ko5ebeHwO-6PraaCm94gKkbNRKOsPJo"

SHARJAH_DB_PATH   = "sharjah.db"
SHARJAH_GDRIVE_ID = "1pOEDcRhh9_SujLRnyL31NbSB9Lo5o65t"

def download_db(db_path, gdrive_id):
    try:
        import gdown
        gdown.download(id=gdrive_id, output=db_path, quiet=False)
    except Exception as e:
        st.error(f"gdown failed: {e}")
        return False
    if not os.path.exists(db_path):
        return False
    with open(db_path, "rb") as f:
        header = f.read(16)
    if not header.startswith(b"SQLite format 3"):
        os.remove(db_path)
        st.error("Downloaded file is not a valid SQLite database.")
        return False
    return True

def ensure_db(db_path, gdrive_id):
    if os.path.exists(db_path):
        with open(db_path, "rb") as f:
            header = f.read(16)
        if not header.startswith(b"SQLite format 3"):
            os.remove(db_path)

    if not os.path.exists(db_path):
        with st.spinner(f"Downloading {db_path} from Google Drive..."):
            success = download_db(db_path, gdrive_id)
        if not success:
            st.stop()

st.sidebar.title("Smart Building")
st.sidebar.markdown("**Energy Anomaly Detection**")
st.sidebar.divider()

dataset = st.sidebar.radio(
    "Dataset",
    ["ASHRAE GEPIII", "Sharjah 2024"],
)
st.sidebar.divider()

groq_key = st.sidebar.text_input("Groq API Key", type="password")

if dataset == "ASHRAE GEPIII":

    ensure_db(GEPIII_DB_PATH, GEPIII_GDRIVE_ID)

    @st.cache_data
    def get_building_list(db_path):
        conn = sqlite3.connect(db_path)
        bids = pd.read_sql("SELECT DISTINCT building_id FROM energy_features", conn)
        conn.close()
        return sorted(bids["building_id"].tolist())

    @st.cache_data
    def load_building_data(db_path, building_id):
        conn = sqlite3.connect(db_path)
        df = pd.read_sql(
            """SELECT building_id, timestamp, meter_reading, rolling_mean_24h,
                      rolling_std_24h, lag_7day, hour, day_of_week, month, is_weekend
               FROM energy_features
               WHERE building_id = ?""",
            conn, params=(int(building_id),)
        )
        try:
            expl = pd.read_sql(
                "SELECT * FROM anomaly_explanations WHERE building_id = ?",
                conn, params=(int(building_id),)
            )
        except Exception:
            expl = pd.DataFrame()
        conn.close()

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["z_score"] = (df["meter_reading"] - df["rolling_mean_24h"]) / df["rolling_std_24h"].replace(0, 1)

        float_cols = df.select_dtypes("float64").columns
        df[float_cols] = df[float_cols].astype("float32")

        return df, expl

    @st.cache_data
    def run_isolation_forest(df):
        FEATURES = ["meter_reading","rolling_mean_24h","rolling_std_24h",
                    "lag_7day","hour","day_of_week","month","is_weekend"]
        df_m   = df[FEATURES + ["building_id","timestamp","z_score"]].dropna().copy()
        scaler = MinMaxScaler()
        X      = scaler.fit_transform(df_m[FEATURES])
        iso    = IsolationForest(n_estimators=200, contamination=0.09,
                                 random_state=42, n_jobs=-1)
        iso.fit(X)
        df_m["if_anomaly"]     = (iso.predict(X) == -1).astype(int)
        df_m["if_score"]       = iso.decision_function(X)
        df_m["zscore_anomaly"] = (df_m["z_score"].abs() > 3).astype(int)
        return df_m

    st.sidebar.markdown("ASHRAE GEPIII | UEAS Potsdam")

    buildings    = get_building_list(GEPIII_DB_PATH)
    sel_building = st.sidebar.selectbox("Select Building", buildings)

    method_map = {"Isolation Forest": "if_anomaly", "Z-score": "zscore_anomaly"}
    sel_method = st.sidebar.selectbox("Detection Method", list(method_map.keys()))
    method_col = method_map[sel_method]

    df, expl_df = load_building_data(GEPIII_DB_PATH, sel_building)

    with st.spinner("Running Isolation Forest..."):
        bdf = run_isolation_forest(df)

    st.title("Smart Building Energy Anomaly Detection")
    st.markdown(f"**Dataset:** ASHRAE GEPIII | **Building {sel_building}** | Method: **{sel_method}**")
    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Readings",    f"{len(bdf):,}")
    c2.metric("Anomalies Flagged", f"{bdf[method_col].sum():,}")
    c3.metric("Anomaly Rate",      f"{bdf[method_col].mean()*100:.1f}%")
    c4.metric("Avg Consumption",   f"{bdf['meter_reading'].mean():.1f} kWh")
    st.divider()

    st.subheader("Consumption with Anomaly Flags")
    plot_df = bdf.set_index("timestamp").head(720)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(plot_df.index, plot_df["meter_reading"], color="steelblue", linewidth=0.8, label="Meter Reading")
    anoms = plot_df[plot_df[method_col] == 1]
    ax.scatter(anoms.index, anoms["meter_reading"], color="red", s=20, zorder=5, label="Anomaly")
    ax.set_ylabel("kWh")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.subheader("Anomaly Rate by Hour of Day")
    hourly = bdf.groupby("hour")[method_col].mean() * 100

    fig2, ax2 = plt.subplots(figsize=(10, 3))
    ax2.bar(hourly.index, hourly.values, color="#7B2FBE", edgecolor="white")
    ax2.set_xlabel("Hour of Day")
    ax2.set_ylabel("Anomaly Rate (%)")
    ax2.set_title(f"Anomaly Rate by Hour — {sel_method}")
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    st.subheader("Anomaly Rate by Day of Week")
    day_labels = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    daily = bdf.groupby("day_of_week")[method_col].mean() * 100

    fig3, ax3 = plt.subplots(figsize=(8, 3))
    ax3.bar(daily.index, daily.values, color="#4A1472", edgecolor="white")
    ax3.set_xticks(range(7))
    ax3.set_xticklabels(day_labels)
    ax3.set_ylabel("Anomaly Rate (%)")
    ax3.set_title(f"Anomaly Rate by Day — {sel_method}")
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()
    st.divider()

    if len(expl_df) > 0:
        st.subheader("Pre-generated LLM Explanations")
        bld_expl = expl_df[expl_df["building_id"] == sel_building]
        if len(bld_expl) > 0:
            for _, row in bld_expl.head(3).iterrows():
                with st.expander(f"{row['timestamp']} — {row['meter_reading']:.1f} kWh"):
                    st.write(row["llm_explanation"])
        else:
            st.info("No pre-generated explanations for this building.")
        st.divider()

    st.subheader("Explain a Specific Anomaly Live")

    anomaly_rows = bdf[bdf[method_col] == 1].reset_index(drop=True)

    if not groq_key:
        st.info("Enter your Groq API key in the sidebar to enable live explanations.")
    elif len(anomaly_rows) == 0:
        st.warning("No anomalies found for this building with the selected method.")
    else:
        sel_idx = st.selectbox(
            "Select anomaly to explain",
            anomaly_rows.index,
            format_func=lambda i: f"{anomaly_rows.loc[i,'timestamp']} — {anomaly_rows.loc[i,'meter_reading']:.1f} kWh"
        )

        if st.button("Explain with Groq", key="gepiii_explain"):
            row    = anomaly_rows.loc[sel_idx]
            client = Groq(api_key=groq_key)

            def safe(val, fmt='.2f'):
                try:
                    return format(float(val), fmt) if pd.notna(val) else 'N/A'
                except:
                    return 'N/A'

            prompt = f"""
You are an energy analyst for a smart building management system.

An anomaly was detected with the following data:
- Building ID      : {row['building_id']}
- Timestamp        : {row['timestamp']}
- Meter Reading    : {safe(row['meter_reading'])} kWh
- 24hr Rolling Mean: {safe(row.get('rolling_mean_24h'))} kWh
- 24hr Rolling Std : {safe(row.get('rolling_std_24h'))} kWh
- Z-score          : {safe(row.get('z_score'))}
- 7-day Lag Value  : {safe(row.get('lag_7day'))} kWh
- Hour of Day      : {safe(row.get('hour'), '.0f')}
- Day of Week      : {safe(row.get('day_of_week'), '.0f')} (0=Mon, 6=Sun)
- Is Weekend       : {'Yes' if row.get('is_weekend') == 1 else 'No'}
- Isolation Forest : {'Yes' if row.get('if_anomaly') == 1 else 'No'}
- Z-score Flag     : {'Yes' if row.get('zscore_anomaly') == 1 else 'No'}

In 3-4 sentences explain: why this reading is anomalous, what the likely cause is,
and what action a building manager should take.
"""

            with st.spinner("Asking Groq..."):
                resp = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.4,
                    max_tokens=200
                )
            st.success("Explanation")
            st.write(resp.choices[0].message.content.strip())

    st.divider()
    st.caption("Smart Building Energy Anomaly Detection | ASHRAE GEPIII | UEAS Potsdam | Dharmik · Hardip · Saurav · Dharmin")

else:

    ensure_db(SHARJAH_DB_PATH, SHARJAH_GDRIVE_ID)

    STREAM_CONFIG = {
        "Power":       {"table": "power_consumption_flagged", "value_col": "power_W",      "group_col": "appliance", "unit": "W"},
        "Temperature": {"table": "temperature_flagged",       "value_col": "temp_C",       "group_col": "location",  "unit": "C"},
        "Humidity":    {"table": "humidity_flagged",          "value_col": "humidity_pct", "group_col": "location",  "unit": "%"},
        "Motion":      {"table": "motion_flagged",            "value_col": "motion_value", "group_col": "location",  "unit": ""},
    }

    @st.cache_data
    def get_group_list(db_path, table, group_col):
        conn = sqlite3.connect(db_path)
        vals = pd.read_sql(f"SELECT DISTINCT {group_col} FROM {table}", conn)
        conn.close()
        return sorted(vals[group_col].tolist())

    @st.cache_data
    def load_sharjah_data(db_path, table, group_col, group_val):
        conn = sqlite3.connect(db_path)
        df = pd.read_sql(f"SELECT * FROM {table} WHERE {group_col} = ?", conn, params=(group_val,))
        conn.close()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        float_cols = df.select_dtypes("float64").columns
        df[float_cols] = df[float_cols].astype("float32")
        return df

    st.sidebar.markdown("Sharjah IoT Smart Building 2024 | UEAS Potsdam")

    sel_stream = st.sidebar.selectbox("Data Stream", list(STREAM_CONFIG.keys()))
    cfg = STREAM_CONFIG[sel_stream]

    groups = get_group_list(SHARJAH_DB_PATH, cfg["table"], cfg["group_col"])
    sel_group = st.sidebar.selectbox(cfg["group_col"].capitalize(), groups)

    method_map = {
        "Consensus (2 of 3)": "anomaly_consensus",
        "Isolation Forest":   "anomaly_iforest",
        "Z-score":            "anomaly_zscore",
        "IQR":                "anomaly_iqr",
    }
    sel_method = st.sidebar.selectbox("Detection Method", list(method_map.keys()))
    method_col = method_map[sel_method]

    sdf = load_sharjah_data(SHARJAH_DB_PATH, cfg["table"], cfg["group_col"], sel_group)
    value_col = cfg["value_col"]
    unit = cfg["unit"]

    st.title("Smart Building Energy Anomaly Detection")
    st.markdown(f"**Dataset:** Sharjah 2024 | **Stream:** {sel_stream} | **{cfg['group_col'].capitalize()}:** {sel_group} | Method: **{sel_method}**")
    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Readings",    f"{len(sdf):,}")
    c2.metric("Anomalies Flagged", f"{int(sdf[method_col].sum()):,}")
    c3.metric("Anomaly Rate",      f"{sdf[method_col].mean()*100:.1f}%")
    c4.metric(f"Avg {sel_stream}", f"{sdf[value_col].mean():.1f} {unit}")
    st.divider()

    st.subheader(f"{sel_stream} with Anomaly Flags")
    plot_df = sdf.set_index("timestamp").head(2000)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(plot_df.index, plot_df[value_col], color="steelblue", linewidth=0.8, label=sel_stream)
    anoms = plot_df[plot_df[method_col] == True]
    ax.scatter(anoms.index, anoms[value_col], color="red", s=20, zorder=5, label="Anomaly")
    ax.set_ylabel(f"{sel_stream} ({unit})" if unit else sel_stream)
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.subheader("Anomaly Rate by Hour of Day")
    sdf["hour"] = sdf["timestamp"].dt.hour
    hourly = sdf.groupby("hour")[method_col].mean() * 100

    fig2, ax2 = plt.subplots(figsize=(10, 3))
    ax2.bar(hourly.index, hourly.values, color="#7B2FBE", edgecolor="white")
    ax2.set_xlabel("Hour of Day")
    ax2.set_ylabel("Anomaly Rate (%)")
    ax2.set_title(f"Anomaly Rate by Hour — {sel_method}")
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    st.subheader("Anomaly Rate by Day of Week")
    day_labels = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    sdf["day_of_week"] = sdf["timestamp"].dt.dayofweek
    daily = sdf.groupby("day_of_week")[method_col].mean() * 100

    fig3, ax3 = plt.subplots(figsize=(8, 3))
    ax3.bar(daily.index, daily.values, color="#4A1472", edgecolor="white")
    ax3.set_xticks(range(7))
    ax3.set_xticklabels(day_labels)
    ax3.set_ylabel("Anomaly Rate (%)")
    ax3.set_title(f"Anomaly Rate by Day — {sel_method}")
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()
    st.divider()

    st.subheader("Explain a Specific Anomaly Live")

    anomaly_rows = sdf[sdf[method_col] == True].reset_index(drop=True)

    if not groq_key:
        st.info("Enter your Groq API key in the sidebar to enable live explanations.")
    elif len(anomaly_rows) == 0:
        st.warning(f"No anomalies found for {sel_group} with the selected method.")
    else:
        sel_idx = st.selectbox(
            "Select anomaly to explain",
            anomaly_rows.index,
            format_func=lambda i: f"{anomaly_rows.loc[i,'timestamp']} — {anomaly_rows.loc[i, value_col]:.1f} {unit}"
        )

        if st.button("Explain with Groq", key="sharjah_explain"):
            row    = anomaly_rows.loc[sel_idx]
            client = Groq(api_key=groq_key)

            def safe(val, fmt='.2f'):
                try:
                    return format(float(val), fmt) if pd.notna(val) else 'N/A'
                except:
                    return 'N/A'

            prompt = f"""
You are an IoT smart-building analyst.

An anomaly was detected with the following data:
- Dataset          : Sharjah 2024 IoT Smart Building
- Stream           : {sel_stream}
- {cfg['group_col'].capitalize()}       : {sel_group}
- Timestamp        : {row['timestamp']}
- Value            : {safe(row[value_col])} {unit}
- Z-score Flag     : {'Yes' if row.get('anomaly_zscore') else 'No'}
- IQR Flag         : {'Yes' if row.get('anomaly_iqr') else 'No'}
- Isolation Forest : {'Yes' if row.get('anomaly_iforest') else 'No'}
- Consensus Flag   : {'Yes' if row.get('anomaly_consensus') else 'No'}

In 3-4 sentences explain: why this reading is anomalous, what the likely cause is,
and what action a building manager should take.
"""

            with st.spinner("Asking Groq..."):
                resp = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.4,
                    max_tokens=200
                )
            st.success("Explanation")
            st.write(resp.choices[0].message.content.strip())

    st.divider()
    st.caption("Smart Building Energy Anomaly Detection | Sharjah 2024 IoT Dataset | UEAS Potsdam | Dharmik · Hardip · Saurav · Dharmin")
