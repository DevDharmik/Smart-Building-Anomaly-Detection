streamlit_code = '''
import streamlit as st
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import os
from groq import Groq

st.set_page_config(page_title="Smart Building Anomaly Detection",
                   page_icon="🏢", layout="wide")

@st.cache_data
def load_data(db_path):
    conn = sqlite3.connect(db_path)
    df   = pd.read_sql("SELECT * FROM energy_features", conn)
    try:
        expl = pd.read_sql("SELECT * FROM anomaly_explanations", conn)
    except:
        expl = pd.DataFrame()
    conn.close()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["z_score"]   = (df["meter_reading"] - df["rolling_mean_24h"]) / \
                       df["rolling_std_24h"].replace(0, 1)
    return df, expl

# ── Sidebar ────────────────────────────────────────────────────────────────
st.sidebar.title("🏢 Smart Building")
st.sidebar.markdown("**Energy Anomaly Detection**")
st.sidebar.markdown("ASHRAE GEPIII | UEAS Potsdam")
st.sidebar.divider()

db_path      = st.sidebar.text_input("SQLite DB path", "smart_building.db")
groq_key     = st.sidebar.text_input("Groq API Key", type="password")
sel_building = None

if os.path.exists(db_path):
    df, expl_df  = load_data(db_path)

    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import MinMaxScaler

    FEATURES = ["meter_reading","rolling_mean_24h","rolling_std_24h",
                "lag_7day","hour","day_of_week","month","is_weekend"]

    df_m     = df[FEATURES + ["building_id","timestamp"]].dropna().copy()
    scaler   = MinMaxScaler()
    X        = scaler.fit_transform(df_m[FEATURES])
    iso      = IsolationForest(n_estimators=200, contamination=0.09,
                               random_state=42, n_jobs=-1)
    iso.fit(X)
    df_m["if_anomaly"]    = (iso.predict(X) == -1).astype(int)
    df_m["zscore_anomaly"] = (df_m["z_score"] if "z_score" in df_m.columns
                              else (df_m["meter_reading"] - df_m["rolling_mean_24h"]) /
                              df_m["rolling_std_24h"].replace(0,1)).abs().gt(3).astype(int) \
                              if "zscore_anomaly" not in df_m.columns else df_m["zscore_anomaly"]
    df_m["z_score"]       = (df_m["meter_reading"] - df_m["rolling_mean_24h"]) / \
                             df_m["rolling_std_24h"].replace(0, 1)

    buildings    = sorted(df_m["building_id"].unique())
    sel_building = st.sidebar.selectbox("Select Building", buildings)
    method_map   = {"Z-score":"zscore_anomaly","Isolation Forest":"if_anomaly"}
    sel_method   = st.sidebar.selectbox("Detection Method", list(method_map.keys()))
    method_col   = method_map[sel_method]

    bdf = df_m[df_m["building_id"] == sel_building].copy()

    # ── Header ─────────────────────────────────────────────────────────────
    st.title("🏢 Smart Building Energy Anomaly Detection")
    st.markdown(f"**Building {sel_building}** | Method: **{sel_method}**")
    st.divider()

    # ── KPIs ───────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Readings",    f"{len(bdf):,}")
    c2.metric("Anomalies Flagged", f"{bdf[method_col].sum():,}")
    c3.metric("Anomaly Rate",      f"{bdf[method_col].mean()*100:.1f}%")
    c4.metric("Avg Consumption",   f"{bdf['meter_reading'].mean():.1f} kWh")
    st.divider()

    # ── Time series ────────────────────────────────────────────────────────
    st.subheader("📈 Consumption with Anomaly Flags")
    plot_df = bdf.set_index("timestamp").head(720)
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(plot_df.index, plot_df["meter_reading"],
            color="steelblue", linewidth=0.8, label="Meter Reading")
    anoms = plot_df[plot_df[method_col] == 1]
    ax.scatter(anoms.index, anoms["meter_reading"],
               color="red", s=20, zorder=5, label="Anomaly")
    ax.set_ylabel("kWh")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── Hourly pattern ─────────────────────────────────────────────────────
    st.subheader("🕐 Anomaly Rate by Hour")
    hourly = bdf.groupby("hour")[method_col].mean() * 100
    fig2, ax2 = plt.subplots(figsize=(10, 3))
    ax2.bar(hourly.index, hourly.values, color="#7B2FBE", edgecolor="white")
    ax2.set_xlabel("Hour of Day")
    ax2.set_ylabel("Anomaly Rate (%)")
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()
    st.divider()

    # ── Pre-generated explanations ─────────────────────────────────────────
    if len(expl_df) > 0:
        st.subheader("🤖 Pre-generated LLM Explanations")
        bld_expl = expl_df[expl_df["building_id"] == sel_building]
        for _, row in bld_expl.head(3).iterrows():
            with st.expander(f"📍 {row['timestamp']} — {row['meter_reading']:.1f} kWh"):
                st.write(row["llm_explanation"])
        st.divider()

    # ── Live Groq ──────────────────────────────────────────────────────────
    st.subheader("⚡ Explain a Specific Anomaly Live")
    anomaly_rows = bdf[bdf[method_col] == 1].reset_index(drop=True)

    if groq_key and len(anomaly_rows) > 0:
        sel_idx = st.selectbox(
            "Select anomaly",
            anomaly_rows.index,
            format_func=lambda i:
                f"{anomaly_rows.loc[i,'timestamp']} — {anomaly_rows.loc[i,'meter_reading']:.1f} kWh"
        )
        if st.button("🔍 Explain with Groq"):
            row    = anomaly_rows.loc[sel_idx]
            client = Groq(api_key=groq_key)
            prompt = f"""
You are an energy analyst. An anomaly was detected:
- Building: {row["building_id"]}
- Time: {row["timestamp"]}
- Reading: {row["meter_reading"]:.2f} kWh
- Rolling Mean: {row.get("rolling_mean_24h", "N/A")}
- Z-score: {row.get("z_score", "N/A")}
- Hour: {int(row.get("hour", 0))}
- Is Weekend: {"Yes" if row.get("is_weekend") == 1 else "No"}

In 3-4 sentences: why anomalous, likely cause, recommended action.
"""
            with st.spinner("Asking Groq..."):
                resp = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.4, max_tokens=200
                )
            st.success("✅ Explanation")
            st.write(resp.choices[0].message.content.strip())
    elif not groq_key:
        st.info("Enter your Groq API key in the sidebar to enable live explanations.")

else:
    st.warning(f"DB not found at: {db_path} — update the path in the sidebar.")

st.divider()
st.caption("Smart Building | ASHRAE GEPIII | UEAS Potsdam | Dharmik · Hardip · Saurav · Dharmin")
'''

with open('/content/drive/MyDrive/app.py', 'w') as f:
    f.write(streamlit_code)

print("✅ app.py saved to Drive")
