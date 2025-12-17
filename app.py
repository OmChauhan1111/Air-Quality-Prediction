# app.py — Air Quality — Smart Predictor (user-friendly UI)
# Improved UI, clearer map controls, coordinate validation, safer pydeck usage

import streamlit as st
import joblib, json, os
from pathlib import Path
import pandas as pd
import numpy as np
from io import BytesIO
import html
import plotly.graph_objects as go
import datetime

# optional pydeck (kept but used only if present)
try:
    import pydeck as pdk
    PYDECK = True
except Exception:
    PYDECK = False

# Config
SAVED_DIR = Path("saved_models")
META_FILE = SAVED_DIR / "model_metadata.json"
PREPROC_FILE = SAVED_DIR / "preprocessor.joblib"
LABEL_ENCODER_FILE = SAVED_DIR / "label_encoder.joblib"

# runtime/persistent storage for history
RUNTIME_DIR = Path("saved_data")
RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
HISTORY_XLSX = RUNTIME_DIR / "prediction_history.xlsx"
HISTORY_CSV = RUNTIME_DIR / "prediction_history.csv"

st.set_page_config(page_title="Air Quality — Smart Predictor", layout="wide")

# ---------- Small visual theme ----------
st.markdown(
    """
    <style>
    .header { background: linear-gradient(90deg, #2b4865 0%, #0f9b0f 100%); padding: 14px; border-radius: 10px; color: white; }
    .card { background: #fff; border-radius:10px; padding:12px; box-shadow:0 6px 18px rgba(0,0,0,0.08); }
    .small-muted { color:#6c757d; font-size:13px; }
    .kpi { background:#f7f9fb; padding:10px; border-radius:8px; text-align:center }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="header"><h1 style="margin:0">Air Quality — Smart Predictor</h1><div class="small-muted">Single predict • Batch CSV • Persistent history • Map view</div></div>', unsafe_allow_html=True)
st.markdown("\n")

# ---------- Helpers ----------

def aqi_to_category(aqi):
    aqi = float(aqi)
    if aqi <= 50: return "Good", "#2E8B57"
    if aqi <= 100: return "Moderate", "#F4D03F"
    if aqi <= 150: return "Unhealthy for Sensitive Groups", "#FF8C00"
    if aqi <= 200: return "Unhealthy", "#FF4500"
    if aqi <= 300: return "Very Unhealthy", "#8B008B"
    return "Hazardous", "#660000"


def pretty_name_from_feature(f):
    s = f.replace('_',' ').replace('.',' ').strip()
    low = f.lower()
    if low in ('pm25','pm2.5','pm_2_5','pm2_5'): return "PM2.5 (µg/m³)"
    if low in ('pm10','pm_10'): return "PM10 (µg/m³)"
    if 'no2' in low: return "NO₂ (µg/m³)"
    if 'so2' in low: return "SO₂ (µg/m³)"
    if 'o3' in low or 'ozone' in low: return "O₃ (µg/m³)"
    if low == 'co' or low.startswith('co'): return "CO (mg/m³)"
    if 'temp' in low: return "Temperature (°C)"
    if 'humid' in low or 'rh' in low: return "Relative Humidity (%)"
    if 'wind' in low: return "Wind speed (m/s)"
    return s.title()


def suggested_range_default(f):
    low = f.lower()
    if any(x in low for x in ['pm2.5','pm25','pm_2_5','pm2_5']): return (0.0,500.0,12.0)
    if any(x in low for x in ['pm10','pm_10']): return (0.0,600.0,40.0)
    if 'no2' in low: return (0.0,400.0,30.0)
    if 'so2' in low: return (0.0,400.0,10.0)
    if 'o3' in low or 'ozone' in low: return (0.0,400.0,20.0)
    if low == 'co' or low.startswith('co'): return (0.0,50.0,0.7)
    if 'temp' in low: return (-30.0,60.0,25.0)
    if 'humid' in low or 'rh' in low: return (0.0,100.0,50.0)
    if 'wind' in low: return (0.0,40.0,3.0)
    return (0.0,1000.0,10.0)


def ensure_numeric(df, cols):
    df2 = df.copy(); ok=[]
    for c in cols:
        df2[c] = pd.to_numeric(df2[c], errors='coerce')
        if df2[c].notna().any(): ok.append(c)
    return df2, ok


def to_csv_bytes(df):
    buf = BytesIO(); df.to_csv(buf,index=False); buf.seek(0); return buf


def df_to_excel_bytes(df):
    buf = BytesIO()
    try:
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="history")
        buf.seek(0)
        return buf.getvalue()
    except Exception:
        # fallback: return CSV bytes if openpyxl not available
        return to_csv_bytes(df).getvalue()


# ---------- Persistence helpers for history ----------

def save_history_to_disk(df):
    """Save DataFrame to XLSX and CSV on disk."""
    try:
        # save xlsx if possible
        try:
            with pd.ExcelWriter(HISTORY_XLSX, engine="openpyxl") as writer:
                df.to_excel(writer, index=False, sheet_name="history")
        except Exception:
            # ignore openpyxl failure and still save CSV
            pass
        # save csv too
        df.to_csv(HISTORY_CSV, index=False)
        return True, None
    except Exception as e:
        return False, str(e)


def load_history_from_disk():
    if HISTORY_CSV.exists():
        try:
            return pd.read_csv(HISTORY_CSV)
        except Exception:
            try:
                return pd.read_excel(HISTORY_XLSX)
            except Exception:
                return None
    if HISTORY_XLSX.exists():
        try:
            return pd.read_excel(HISTORY_XLSX)
        except Exception:
            return None
    return None


# ---------- Load model metadata and saved objects ----------
if not META_FILE.exists():
    st.error("Model metadata not found. Place saved_models/model_metadata.json in project folder.")
    st.stop()

meta = json.load(open(META_FILE))
saved = meta.get("saved_models", {})

def pick_best_entry(saved):
    if saved.get("regression"):
        pref = ["random_forest_regressor","decision_tree_regressor","linear_regression"]
        for p in pref:
            if p in saved["regression"]: return ("regression", p, saved["regression"][p])
        k = list(saved["regression"].keys())[0]; return ("regression", k, saved["regression"][k])
    if saved.get("classification"):
        pref = ["knn_classifier","random_forest_classifier","decision_tree_classifier","logistic_regression"]
        for p in pref:
            if p in saved["classification"]: return ("classification", p, saved["classification"][p])
        k = list(saved["classification"].keys())[0]; return ("classification", k, saved["classification"][k])
    return (None, None, None)

task, model_key, model_path = pick_best_entry(saved)
st.sidebar.markdown("### Model info")
st.sidebar.write("Task:", task)
st.sidebar.write("Model:", model_key)
st.sidebar.write("Loaded:", "✅" if Path(model_path).exists() else "❌")
st.sidebar.markdown("---")

# Load model/preproc/label encoder (non-fatal)
model = None; preproc = None; label_encoder = None
try: model = joblib.load(model_path)
except Exception: model = None
if PREPROC_FILE.exists():
    try: preproc = joblib.load(PREPROC_FILE)
    except Exception: preproc = None
if LABEL_ENCODER_FILE.exists():
    try: label_encoder = joblib.load(LABEL_ENCODER_FILE)
    except Exception: label_encoder = None

# Determine features
if task == "regression": features = meta.get("features_regression") or meta.get("features") or []
else: features = meta.get("features_classification") or meta.get("features") or []

if not features:
    st.warning("Feature list missing in metadata. App will show generic inputs but predictions may fail.")
    features = []

# Init history
if "prediction_history" not in st.session_state:
    st.session_state["prediction_history"] = []

# load persisted history if exists (only if session empty)
disk_hist = load_history_from_disk()
if disk_hist is not None and not st.session_state.get("prediction_history"):
    try:
        st.session_state["prediction_history"] = disk_hist.to_dict(orient="records")
        st.success(f"Loaded {len(st.session_state['prediction_history'])} rows from saved_data/")
    except Exception:
        pass

# ---------- Main UI (inputs + upload) ----------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Single prediction")
st.markdown("Enter sensor readings and click **Predict Now**.")
inputs = {}
cols = st.columns(2)
defaults = {}
for i, f in enumerate(features):
    label = pretty_name_from_feature(f)
    lo, hi, default = suggested_range_default(f)
    safe_key = "fld_" + "".join(c if c.isalnum() else "_" for c in f)
    defaults[safe_key] = float(default)
    if safe_key not in st.session_state: st.session_state[safe_key] = float(default)
    col = cols[i % 2]
    if hi <= 1000:
        val = col.slider(label, min_value=float(lo), max_value=float(hi),
                         value=st.session_state[safe_key], step=(hi-lo)/100 if (hi-lo) > 0 else 1.0, key=safe_key)
    else:
        val = col.number_input(label, min_value=float(lo), max_value=float(hi), value=st.session_state[safe_key], key=safe_key, format="%.6f")
    inputs[f] = val
st.markdown("</div>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1,1,1])
with col1:
    predict_btn = st.button("Predict Now")
with col2:
    reset_btn = st.button("Reset to defaults")
with col3:
    upload = st.file_uploader("Upload CSV for batch prediction", type=["csv"]) 

if reset_btn:
    for k,v in defaults.items(): st.session_state[k] = v
    st.success("Inputs reset to defaults (session).")

# ---------- CSV upload & batch predict (minimal mapping) ----------
uploaded_df = None
if upload is not None:
    try:
        df = pd.read_csv(upload)
        st.markdown("**Preview (first 5 rows)**")
        st.dataframe(df.head(5))

        # auto-rename helper
        def auto_rename_cols(df, expected):
            mapr = {}
            for c in df.columns:
                lowc = c.lower().replace(" ","" ).replace("_","" ).replace("-","" ).replace(".","" )
                for f in expected:
                    lowf = f.lower().replace(" ","" ).replace("_","" ).replace("-","" ).replace(".","" )
                    if lowc == lowf or (lowc.replace("2","") == lowf.replace("2","") and ("pm" in lowf or "pm" in lowc)):
                        mapr[c] = f; break
            if mapr: df = df.rename(columns=mapr)
            return df, mapr

        df, mapr = auto_rename_cols(df, features)
        if mapr: st.success("Auto-rename applied: " + str(mapr))

        # mapping UI if required
        missing = set(features) - set(df.columns)
        if missing:
            st.warning(f"CSV missing columns required by model: {missing}. Please map them below.")
            csv_cols = list(df.columns)
            mapping = {}
            for f in features:
                mapping[f] = st.selectbox(f"Map CSV column to feature '{f}'", options=["<none>"]+csv_cols, key="map_"+f)
            if all(mapping[f] != "<none>" for f in features):
                try:
                    X = df[[mapping[f] for f in features]].copy(); X.columns = features
                except Exception as e:
                    st.error("Failed to build X from mapping: " + str(e)); X = None
            else: X = None
        else:
            X = df[features].copy()

        if X is not None:
            Xc, ok = ensure_numeric(X, X.columns.tolist())
            try: X_in = preproc.transform(Xc) if preproc is not None else Xc
            except Exception: X_in = Xc
            try:
                preds = model.predict(X_in) if model is not None else None
            except Exception as e:
                st.error("Model prediction failed: " + str(e)); preds = None

            if preds is not None:
                # attach preds
                try:
                    df["aqi_pred_numeric"] = [float(x) for x in preds]
                    df["aqi_pred_category"] = [aqi_to_category(float(x))[0] for x in preds]
                except Exception:
                    if label_encoder is not None:
                        try: df["predicted_label"] = label_encoder.inverse_transform(preds)
                        except Exception: df["predicted_label"] = preds
                    else: df["predicted_label"] = preds

                st.markdown("**Predictions (first 10 rows)**")
                st.dataframe(df.head(10))
                st.download_button("Download predictions CSV", data=to_csv_bytes(df), file_name="predictions.csv")
                uploaded_df = df.copy()

                # add rows to session history
                try:
                    for _, row in df.iterrows():
                        entry = {"time": row.get("timestamp") or row.get("date") or pd.Timestamp.now().isoformat()}
                        for f in features: entry[f] = row.get(f)
                        if 'aqi_pred_numeric' in row.index: entry['aqi_pred'] = float(row['aqi_pred_numeric'])
                        else:
                            if 'predicted_label' in row.index: entry['label_pred'] = row['predicted_label']
                        st.session_state["prediction_history"].append(entry)
                except Exception:
                    pass

                # persist history
                try:
                    hist_df = pd.DataFrame(st.session_state["prediction_history"])
                    ok, err = save_history_to_disk(hist_df)
                    if ok: st.success("Saved history to saved_data/")
                except Exception:
                    pass

            else:
                st.error("Prediction couldn't be made (no model).")
        else:
            st.info("Provide mapping for all features to run batch prediction.")
    except Exception as e:
        st.error("Failed to read uploaded CSV: " + str(e))

# ---------- Single prediction ----------
if predict_btn:
    if not features:
        st.error("Feature list missing, cannot run single prediction.")
    else:
        X_new = pd.DataFrame([inputs], columns=features)
        try: X_in = preproc.transform(X_new) if preproc is not None else X_new
        except Exception as e: st.error("Preprocessor error: " + str(e)); X_in = None
        if X_in is not None:
            try: preds = model.predict(X_in) if model is not None else None
            except Exception as e: st.error("Model predict failed: " + str(e)); preds = None
            if preds is None:
                st.error("No model loaded. Place model in saved_models.")
            else:
                pred0 = preds[0]
                is_numeric = False; numeric_val = None
                try: numeric_val = float(pred0); is_numeric = True
                except Exception: is_numeric = False

                st.markdown('<div class="card">', unsafe_allow_html=True)
                if is_numeric:
                    aqi_val = numeric_val
                    category, color = aqi_to_category(aqi_val)
                    st.markdown(f"<div style='display:flex; gap:12px; align-items:center;'>"
                                f"<div style='background:{color}; padding:14px; border-radius:10px; color:white; font-weight:700;'>AQI {aqi_val:.1f}</div>"
                                f"<div><strong>{category}</strong><div class='small-muted'>Regression result</div></div>"
                                f"</div>", unsafe_allow_html=True)
                    hist_entry = {"time": pd.Timestamp.now().isoformat()}
                    for f in features: hist_entry[f] = inputs[f]
                    hist_entry["aqi_pred"] = float(aqi_val)
                    st.session_state["prediction_history"].append(hist_entry)
                else:
                    label_out = pred0
                    if label_encoder is not None:
                        try: label_out = label_encoder.inverse_transform([pred0])[0]
                        except Exception: label_out = str(pred0)
                    label_str = str(label_out).lower()
                    color = "#6c757d"
                    if "good" in label_str: color="#2E8B57"
                    elif "moderate" in label_str: color="#F4D03F"
                    elif "unhealthy" in label_str: color="#FF4500"
                    st.markdown(f"<div style='display:flex; gap:12px; align-items:center;'>"
                                f"<div style='background:{color}; padding:14px; border-radius:10px; color:white; font-weight:700;'>{html.escape(label_out)}</div>"
                                f"<div><strong>Classification result</strong><div class='small-muted'>Label output</div></div>"
                                f"</div>", unsafe_allow_html=True)
                    hist_entry = {"time": pd.Timestamp.now().isoformat()}
                    for f in features: hist_entry[f] = inputs[f]
                    hist_entry["label_pred"] = str(label_out)
                    st.session_state["prediction_history"].append(hist_entry)
                st.markdown("</div>", unsafe_allow_html=True)

                # save history after single prediction
                try:
                    hist_df = pd.DataFrame(st.session_state["prediction_history"])
                    ok, err = save_history_to_disk(hist_df)
                    if ok: st.success("Saved history to saved_data/")
                except Exception:
                    pass

# ---------- History & visuals ----------
st.markdown("---")
st.subheader("Prediction history (session & persisted)")

hist = st.session_state.get("prediction_history", [])
if hist:
    hdf = pd.DataFrame(hist)
    if 'time' not in hdf.columns: hdf['time'] = pd.Timestamp.now().isoformat()

    # derive categories and colors
    if 'aqi_pred' in hdf.columns:
        try:
            hdf['aqi_pred'] = pd.to_numeric(hdf['aqi_pred'], errors='coerce')
            hdf['pred_category'] = hdf['aqi_pred'].apply(lambda x: aqi_to_category(x)[0] if pd.notna(x) else None)
            hdf['pred_color'] = hdf['aqi_pred'].apply(lambda x: aqi_to_category(x)[1] if pd.notna(x) else '#6c757d')
        except Exception:
            hdf['pred_category'] = None; hdf['pred_color'] = '#6c757d'
    elif 'label_pred' in hdf.columns or 'predicted_label' in hdf.columns:
        lab_col = 'label_pred' if 'label_pred' in hdf.columns else 'predicted_label'
        hdf['pred_category'] = hdf[lab_col].astype(str)
        def label_color(s):
            s = str(s).lower()
            if 'good' in s: return '#2E8B57'
            if 'moderate' in s: return '#F4D03F'
            if 'unhealthy' in s: return '#FF4500'
            if 'very' in s or 'hazard' in s or 'hazardous' in s: return '#660000'
            return '#6c757d'
        hdf['pred_color'] = hdf['pred_category'].apply(label_color)
    else:
        hdf['pred_category'] = None; hdf['pred_color'] = '#6c757d'

    # KPI row
    k1, k2, k3, k4 = st.columns([1,1,1,1])
    total = len(hdf)
    avg_aqi = None
    if 'aqi_pred' in hdf.columns:
        try: avg_aqi = hdf['aqi_pred'].dropna().astype(float).mean()
        except Exception: avg_aqi = None
    k1.metric("Total predictions", total)
    k2.metric("Avg predicted AQI", f"{avg_aqi:.1f}" if avg_aqi is not None else "—")
    latest = hdf.iloc[-1]
    if 'aqi_pred' in latest.index and pd.notna(latest['aqi_pred']):
        k3.metric("Latest", f"AQI {float(latest['aqi_pred']):.1f} — {aqi_to_category(float(latest['aqi_pred']))[0]}")
    elif 'label_pred' in latest.index or 'predicted_label' in latest.index:
        k3.metric("Latest", latest.get('label_pred') or latest.get('predicted_label'))
    else:
        k3.metric("Latest", "—")
    top = hdf['pred_category'].mode().iloc[0] if hdf['pred_category'].notna().any() else '—'
    k4.metric("Most frequent", top)

    st.dataframe(hdf.tail(50))

    # download / clear with confirmation checkbox
    da, db, dc = st.columns([1,1,1])
    with da:
        excel_bytes = df_to_excel_bytes(hdf)
        st.download_button("Download history (Excel/CSV)", data=excel_bytes, file_name="prediction_history.xlsx")
    with db:
        st.download_button("Download history (CSV)", data=to_csv_bytes(hdf), file_name="prediction_history.csv")
    with dc:
        confirm_clear = st.checkbox("I confirm I want to clear history (session + disk)")
        if st.button("Clear history") and confirm_clear:
            st.session_state["prediction_history"] = []
            try:
                if HISTORY_CSV.exists(): HISTORY_CSV.unlink()
                if HISTORY_XLSX.exists(): HISTORY_XLSX.unlink()
                st.success("History cleared.")
            except Exception as e:
                st.warning("Cleared session but failed to delete files: " + str(e))

    # Visuals: pie + timeline + colored timeline
    st.markdown("### Visual summary")
    try:
        cat_counts = hdf['pred_category'].fillna('Unknown').value_counts()
        fig_pie = go.Figure(data=[go.Pie(labels=cat_counts.index.tolist(), values=cat_counts.values.tolist(), hole=0.35)])
        fig_pie.update_layout(title_text='Predicted category distribution', height=300)
        st.plotly_chart(fig_pie, use_container_width=True)
    except Exception:
        pass

    # timeline
    try:
        if 'time' in hdf.columns:
            hdf['time_parsed'] = pd.to_datetime(hdf['time'], errors='coerce')
            tseries = hdf.dropna(subset=['time_parsed']).copy()
            if not tseries.empty:
                fig_t = go.Figure()
                fig_t.add_trace(go.Scatter(x=tseries['time_parsed'], y=tseries.get('aqi_pred', [None]*len(tseries)), mode='lines+markers', marker=dict(color=tseries['pred_color'], size=9), hovertext=tseries['pred_category']))
                fig_t.update_layout(title='Timeline — colored by category', height=350)
                st.plotly_chart(fig_t, use_container_width=True)
    except Exception:
        pass

    # quick plots
    st.markdown("**Quick plots from history**")
    opts = [c for c in hdf.columns if c not in ('time','time_parsed')]
    sel = st.selectbox("Choose column to visualize", options=opts)
    plot_type = st.radio("Plot type", options=["Histogram / Distribution", "Timeseries (by time)"])
    if plot_type == "Histogram / Distribution":
        try:
            vals = pd.to_numeric(hdf[sel], errors='coerce').dropna()
            if not vals.empty:
                fig2 = go.Figure(); fig2.add_trace(go.Histogram(x=vals))
                fig2.update_layout(title=f"Distribution of {sel}", height=320)
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("No numeric values for selected column to plot histogram.")
        except Exception as e:
            st.error("Plot failed: " + str(e))
    else:
        if 'time_parsed' not in hdf.columns: hdf['time_parsed'] = pd.to_datetime(hdf['time'], errors='coerce')
        ts2 = hdf.dropna(subset=['time_parsed']).copy()
        try:
            ts2[sel] = pd.to_numeric(ts2[sel], errors='coerce')
            ts2 = ts2.dropna(subset=[sel])
            if not ts2.empty:
                ts2 = ts2.sort_values('time_parsed')
                fig3 = go.Figure(); fig3.add_trace(go.Scatter(x=ts2['time_parsed'], y=ts2[sel], mode='lines+markers'))
                fig3.update_layout(title=f"{sel} over time", height=320)
                st.plotly_chart(fig3, use_container_width=True)
            else:
                st.info("No numeric data available for timeseries plot of selected column.")
        except Exception as e:
            st.error("Timeseries plot failed: " + str(e))

else:
    st.info("No prediction history yet - do single or batch prediction to populate it.")

# ---------- Map (interactive) ----------
if uploaded_df is not None and len(uploaded_df.columns) > 0:
    st.markdown("---")
    st.subheader("Map (from uploaded CSV)")
    csv_cols = list(uploaded_df.columns)

    with st.expander("Map settings / column selectors", expanded=True):
        lat_col = st.selectbox("Latitude column", options=["<none>"] + csv_cols, index=csv_cols.index('lat') if 'lat' in csv_cols else 0)
        lon_col = st.selectbox("Longitude column", options=["<none>"] + csv_cols, index=csv_cols.index('lon') if 'lon' in csv_cols else (1 if len(csv_cols)>1 else 0))
        extra_hover_cols = st.multiselect("Columns to show in popup/hover", options=csv_cols, default=[c for c in ['aqi_pred','aqi_pred_numeric','predicted_label','pred_category'] if c in csv_cols])
        # show only numeric columns for size selector to avoid misuse
        numeric_cols = [c for c in csv_cols if pd.api.types.is_numeric_dtype(uploaded_df[c])]
        size_by = st.selectbox("Marker size by", options=["Constant"] + numeric_cols, index=0)
        st.write("\n")
        if st.button("Validate coordinates"):
            # quick validation
            if lat_col == "<none>" or lon_col == "<none>":
                st.error("Choose latitude and longitude columns first.")
            else:
                tmp = uploaded_df.copy()
                tmp[lat_col] = pd.to_numeric(tmp[lat_col], errors='coerce')
                tmp[lon_col] = pd.to_numeric(tmp[lon_col], errors='coerce')
                total_rows = len(tmp)
                valid = tmp.dropna(subset=[lat_col, lon_col])
                st.info(f"Total rows: {total_rows}, Valid coordinate rows: {len(valid)}")
                if not valid.empty:
                    st.write(valid[[lat_col, lon_col]].head(10))
                    lat_min, lat_max = valid[lat_col].min(), valid[lat_col].max()
                    lon_min, lon_max = valid[lon_col].min(), valid[lon_col].max()
                    st.write(f"Latitude range: {lat_min} to {lat_max}")
                    st.write(f"Longitude range: {lon_min} to {lon_max}")
                    if not ((-90 <= lat_min <= 90) and (-90 <= lat_max <= 90)):
                        st.warning("Some latitude values are out of valid range (-90 to 90). Check if lat/lon are swapped.")
                    if not ((-180 <= lon_min <= 180) and (-180 <= lon_max <= 180)):
                        st.warning("Some longitude values are out of valid range (-180 to 180). Check if lat/lon are swapped.")

    # render map if columns selected
    if lat_col != "<none>" and lon_col != "<none>":
        try:
            dfmap = uploaded_df.dropna(subset=[lat_col, lon_col]).copy()
            dfmap[lat_col] = pd.to_numeric(dfmap[lat_col], errors='coerce')
            dfmap[lon_col] = pd.to_numeric(dfmap[lon_col], errors='coerce')
            dfmap = dfmap.dropna(subset=[lat_col, lon_col])

            if dfmap.empty:
                st.warning("No valid lat/lon rows found after coercion to numeric.")
            else:
                # derive category/color columns safely
                if 'pred_category' not in dfmap.columns and 'aqi_pred' in dfmap.columns:
                    try:
                        dfmap['pred_category'] = dfmap['aqi_pred'].apply(lambda x: aqi_to_category(float(x))[0] if pd.notna(x) else None)
                        dfmap['pred_color'] = dfmap['aqi_pred'].apply(lambda x: aqi_to_category(float(x))[1] if pd.notna(x) else '#6c757d')
                    except Exception:
                        dfmap['pred_category'] = None; dfmap['pred_color'] = '#6c757d'
                elif 'pred_color' not in dfmap.columns:
                    dfmap['pred_color'] = '#6c757d'

                            # marker size scaling (always big for visibility)
                if size_by != 'Constant' and size_by in dfmap.columns:
                    dfmap['_size'] = pd.to_numeric(dfmap[size_by], errors='coerce').fillna(0)
                    smin, smax = dfmap['_size'].min(), dfmap['_size'].max()
                    if smax > smin:
                        dfmap['_size_scaled'] = 30 + 70 * (dfmap['_size'] - smin) / (smax - smin)
                    else:
                        dfmap['_size_scaled'] = 50
                else:
                    dfmap['_size_scaled'] = 40


                # hover text
                def build_hover(row):
                    parts = []
                    for c in extra_hover_cols:
                        parts.append(f"{c}: {row.get(c)}")
                    return '<br>'.join(parts)
                dfmap['_hover'] = dfmap.apply(build_hover, axis=1)

                # compute center/zoom heuristic
                lat_mean = float(dfmap[lat_col].mean()); lon_mean = float(dfmap[lon_col].mean())
                lat_range = dfmap[lat_col].max() - dfmap[lat_col].min(); lon_range = dfmap[lon_col].max() - dfmap[lon_col].min()
                max_range = max(lat_range, lon_range)
                if max_range <= 0.01: zoom = 12
                elif max_range <= 0.1: zoom = 10
                elif max_range <= 1: zoom = 8
                elif max_range <= 5: zoom = 6
                else: zoom = 3

                # safe pydeck usage: compute numeric rgb columns
                def hex_to_rgb_int(h):
                    try:
                        h = str(h)
                        if h.startswith('#') and len(h) == 7:
                            return int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)
                    except Exception:
                        pass
                    return 100, 100, 100

                rgb = dfmap['pred_color'].apply(lambda x: hex_to_rgb_int(x))
                dfmap['pred_r'] = rgb.apply(lambda t: t[0]); dfmap['pred_g'] = rgb.apply(lambda t: t[1]); dfmap['pred_b'] = rgb.apply(lambda t: t[2])

                # position for pydeck
                dfmap['__pos'] = dfmap.apply(lambda r: [float(r[lon_col]), float(r[lat_col])], axis=1)

                if PYDECK:
                    get_radius_accessor = '_size_scaled' if '_size_scaled' in dfmap.columns else None
                    layer = pdk.Layer(
                        'ScatterplotLayer',
                        data=dfmap,
                        get_position='__pos',
                        get_fill_color='[pred_r, pred_g, pred_b]',
                        get_radius=(get_radius_accessor if get_radius_accessor else 500),
                        pickable=True,
                        auto_highlight=True,
                    )
                    view = pdk.ViewState(latitude=lat_mean, longitude=lon_mean, zoom=zoom)
                    deck = pdk.Deck(layers=[layer], initial_view_state=view)
                    st.pydeck_chart(deck)
                    st.info("Hover on markers for details. Use the 'Validate coordinates' button to inspect raw values.")
                else:
                    fig = go.Figure()
                    fig.add_trace(go.Scattergeo(
                        lat=dfmap[lat_col], lon=dfmap[lon_col], text=dfmap['_hover'], hoverinfo='text',
                        marker=dict(size=dfmap['_size_scaled'], color=['rgb('+','.join(map(str,[r,g,b]))+')' for r,g,b in zip(dfmap['pred_r'], dfmap['pred_g'], dfmap['pred_b'])], line=dict(width=0.3, color='black'))
                    ))
                    fig.update_layout(geo=dict(scope='world', projection_type='natural earth', center=dict(lat=lat_mean, lon=lon_mean)), height=500, title='Map (markers colored by predicted category)')
                    st.plotly_chart(fig, use_container_width=True)
                    st.info("Hover markers to see details. Use the map settings to change which columns are used for lat/lon and popup text.")
        except Exception as e:
            st.error("Map render failed: " + str(e))

# ---------- Footer (friendly tips) ----------
st.markdown("---")
st.markdown(
    """
    **Tips & next steps**
    - Keep `model_metadata.json` in `saved_models/` so the app knows which model to load.
    - If you used scalers or encoders, include `preprocessor.joblib` and `label_encoder.joblib` in `saved_models/`.
    - History is saved in `saved_data/` as CSV (and XLSX if environment supports it).
    - If you deploy on ephemeral services (Streamlit Cloud / free tiers), connect to S3/Google Drive/database for persistence.

    *If you want, I can:* add category filters on the map, cluster nearby points, or enable clicking a marker to open the full row data.
    """,
    unsafe_allow_html=True,
)
