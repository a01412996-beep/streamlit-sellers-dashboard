import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =====================================================
# 1. BASIC CONFIGURATION
# =====================================================

st.set_page_config(page_title="Whirlpool Utility Model", layout="wide")
st.title("Whirlpool Utility Prediction Dashboard")

st.markdown(
    """
This dashboard trains a **Gradient Boosting model** to predict utility based on the 
Whirlpool dataset and allows filtering by Trading Partner, SKU, and Date.
"""
)

# =====================================================
# 2. LOAD & CLEAN DATA
# =====================================================

@st.cache_data
def load_data(csv_path: str) -> pd.DataFrame:
    """Load the CSV file, preprocess fields, and create derived columns."""

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        st.error(
            f"CSV file not found: `{csv_path}`. "
            "Ensure the file is in the same folder as ML_Model.py."
        )
        return pd.DataFrame()

    # Parse DATE column if present
    if "DATE" in df.columns:
        df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")

    # Filter invalid quantities
    if "QTY" in df.columns:
        df = df[df["QTY"] > 0].copy()
    else:
        st.error("Column 'QTY' not found in dataset.")
        return pd.DataFrame()

    # Unit price
    if "GROSS_SALES" in df.columns:
        df["unit_price"] = df["GROSS_SALES"] / df["QTY"]
    else:
        st.error("Column 'GROSS_SALES' not found.")
        return pd.DataFrame()

    # Utility per record
    cost_col = "total_cost_per_unit (AI)"
    if cost_col in df.columns:
        df["UTILITY"] = (df["unit_price"] - df[cost_col]) * df["QTY"]
    elif "UTILITY" not in df.columns:
        st.error(
            f"Neither '{cost_col}' nor an existing 'UTILITY' column was found."
        )
        return pd.DataFrame()

    # Encode category columns
    for col in ["SKU", "SKU_ORIG_TP"]:
        if col in df.columns:
            df[col + "_CODE"] = df[col].astype("category").cat.codes

    # Process Capacity column
    if "Capacidad" in df.columns:
        if df["Capacidad"].dtype == "O":
            cap_num = df["Capacidad"].astype(str).str.extract(r"(\d+\.?\d*)")[0]
            df["Capacidad_NUM"] = pd.to_numeric(cap_num, errors="coerce")
        else:
            df["Capacidad_NUM"] = df["Capacidad"]
    else:
        df["Capacidad_NUM"] = np.nan

    return df


DATA_PATH = "Final_Dataset_4_3.csv"  # Change name if needed
final = load_data(DATA_PATH)

if final.empty:
    st.stop()

st.subheader("Dataset Overview")
st.write(f"Rows loaded: **{len(final):,}**")
st.dataframe(final.head())

# =====================================================
# 3. MODEL TRAINING
# =====================================================

@st.cache_resource
def train_utility_model(df: pd.DataFrame):
    """Train a GradientBoosting regressor to predict UTILITY."""

    features = []

    base_features = [
        "unit_price",
        "COST_UNITARIO",
        "freight_warehouse_cost",
        "admin_cost",
        "total_cost_per_unit (AI)",
        "product_cost",
        "Capacidad_NUM",
        "branch_avg_qty (AI)",
        "QTY",
    ]
    for col in base_features:
        if col in df.columns:
            features.append(col)

    for col in ["SKU_CODE", "SKU_ORIG_TP_CODE"]:
        if col in df.columns:
            features.append(col)

    if "UTILITY" not in df.columns:
        st.error("Column 'UTILITY' not found.")
        return None, None, None, None, None, None

    if len(features) == 0:
        st.error("No usable numeric features found to train the model.")
        return None, None, None, None, None, None

    X = df[features].copy()
    y = df["UTILITY"].copy()

    # Remove rows with missing values
    valid_mask = (~X.isna().any(axis=1)) & (~y.isna())
    X = X[valid_mask]
    y = y[valid_mask]

    if len(X) < 50:
        st.error("Not enough valid rows (<50) to train the model.")
        return None, None, None, None, None, None

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2": r2_score(y_test, y_pred),
    }

    return model, features, X_test, y_test, y_pred, metrics


st.subheader("Model Training")

try:
    model, FEATURES, X_test_u, y_test_u, y_pred_u, metrics_u = train_utility_model(final)
except Exception as e:
    st.error(f"Model training error: {e}")
    st.stop()

if model is None:
    st.stop()

col_m1, col_m2, col_m3 = st.columns(3)
with col_m1:
    st.metric("MAE", f"{metrics_u['MAE']:,.0f} MXN")
with col_m2:
    st.metric("RMSE", f"{metrics_u['RMSE']:,.0f} MXN")
with col_m3:
    st.metric("R²", f"{metrics_u['R2']:.3f}")

# =====================================================
# 4. SIDEBAR FILTERS
# =====================================================

st.sidebar.header("Filters")

# TP filter
if "TP" in final.columns:
    tp_list = sorted(final["TP"].dropna().unique())
    selected_tp = st.sidebar.multiselect("Trading Partner (TP)", tp_list, default=tp_list)
else:
    selected_tp = None

# SKU filter
if "SKU" in final.columns:
    sku_list = sorted(final["SKU"].dropna().unique())
    default_skus = sku_list[:20] if len(sku_list) > 20 else sku_list
    selected_sku = st.sidebar.multiselect("SKU", sku_list, default=default_skus)
else:
    selected_sku = None

# Date filter
if "DATE" in final.columns:
    min_date = final["DATE"].min()
    max_date = final["DATE"].max()
    date_range = st.sidebar.date_input("Date Range", [min_date, max_date])
    start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
else:
    start_date = end_date = None

# Apply filters
mask = pd.Series(True, index=final.index)

if selected_tp is not None:
    mask &= final["TP"].isin(selected_tp)

if selected_sku is not None:
    mask &= final["SKU"].isin(selected_sku)

if start_date is not None and "DATE" in final.columns:
    mask &= (final["DATE"] >= start_date) & (final["DATE"] <= end_date)

filtered = final[mask].copy()

st.subheader("Filtered Data")
st.write(f"Rows after filtering: **{len(filtered):,}**")
st.dataframe(filtered.head(50))

# =====================================================
# 5. PREDICTIONS
# =====================================================

if filtered.empty:
    st.warning("No records match the selected filters.")
else:
    X_filtered = filtered[FEATURES].copy()
    valid_mask_f = ~X_filtered.isna().any(axis=1)

    preds = np.full(len(X_filtered), np.nan)

    if valid_mask_f.any():
        preds[valid_mask_f.values] = model.predict(X_filtered[valid_mask_f])

    filtered["Predicted_UTILITY"] = preds

    # KPIs
    col_k1, col_k2, col_k3 = st.columns(3)
    with col_k1:
        st.metric("Total Real Utility", f"{filtered['UTILITY'].sum():,.0f} MXN")
    with col_k2:
        st.metric(
            "Total Predicted Utility",
            f"{filtered['Predicted_UTILITY'].sum(skipna=True):,.0f} MXN",
        )
    with col_k3:
        st.metric("Valid Predictions", f"{np.isfinite(preds).sum():,}")

    # =================================================
    # 6. PLOTS
    # =================================================

    st.subheader("Model Performance (Test Set)")

    fig1, ax1 = plt.subplots(figsize=(6, 6))
    ax1.scatter(y_test_u, y_pred_u, alpha=0.5)
    min_val = min(y_test_u.min(), y_pred_u.min())
    max_val = max(y_test_u.max(), y_pred_u.max())
    ax1.plot([min_val, max_val], [min_val, max_val], "r--")
    ax1.set_xlabel("True Utility")
    ax1.set_ylabel("Predicted Utility")
    ax1.set_title("Gradient Boosting — Test Set")
    ax1.grid(True)
    st.pyplot(fig1)

    st.subheader("Filtered Predictions")

    valid_rows = filtered[np.isfinite(filtered["Predicted_UTILITY"])]

    if not valid_rows.empty:
        fig2, ax2 = plt.subplots(figsize=(6, 6))
        ax2.scatter(valid_rows["UTILITY"], valid_rows["Predicted_UTILITY"], alpha=0.5)
        min_val_f = min(
            valid_rows["UTILITY"].min(),
            valid_rows["Predicted_UTILITY"].min(),
        )
        max_val_f = max(
            valid_rows["UTILITY"].max(),
            valid_rows["Predicted_UTILITY"].max(),
        )
        ax2.plot([min_val_f, max_val_f], [min_val_f, max_val_f], "r--")
        ax2.set_xlabel("True Utility")
        ax2.set_ylabel("Predicted Utility")
        ax2.set_title("Predictions for Selected Filters")
        ax2.grid(True)
        st.pyplot(fig2)
    else:
        st.info("No valid rows available for plotting.")

    # Table
    st.subheader("Utility Details (First 100 Rows)")
    show_cols = [
        col
        for col in [
            "DATE",
            "TP",
            "SKU",
            "QTY",
            "unit_price",
            "UTILITY",
            "Predicted_UTILITY",
        ]
        if col in filtered.columns
    ]
    st.dataframe(filtered[show_cols].head(100))
