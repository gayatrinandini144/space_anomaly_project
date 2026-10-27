"""
utils/feature_engineering.py
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

SENSOR_COLS = [
    "solar_panel_voltage","battery_charge","power_consumption",
    "internal_temp","external_temp","thermal_control_power",
    "fuel_pressure","thruster_temp","delta_v",
    "signal_strength","data_rate","bit_error_rate",
]

def handle_missing_noisy_data(df):
    df = df.copy()
    cols = [c for c in SENSOR_COLS if c in df.columns]
    df[cols] = df[cols].ffill(limit=3)
    for col in cols:
        mean, std = df[col].mean(), df[col].std()
        df[col] = df[col].clip(mean-5*std, mean+5*std)
    return df

def add_rolling_features(df, windows=(5,10)):
    df = df.copy()
    cols = [c for c in SENSOR_COLS if c in df.columns]
    for col in cols:
        for w in windows:
            df[f"{col}_rmean_{w}"] = df[col].rolling(w,min_periods=1).mean()
            df[f"{col}_rstd_{w}"]  = df[col].rolling(w,min_periods=1).std().fillna(0)
    return df

def add_diff_features(df):
    df = df.copy()
    cols = [c for c in SENSOR_COLS if c in df.columns]
    for col in cols:
        df[f"{col}_diff"] = df[col].diff().fillna(0)
    return df

def get_feature_matrix(df):
    df = add_rolling_features(df)
    df = add_diff_features(df)
    skip = {"timestamp","anomaly_label","rul","anomaly_type"}
    feature_cols = [c for c in df.columns if c not in skip
                    and df[c].dtype in [np.float64,np.int64,float,int]]
    X = df[feature_cols].fillna(0).values
    y = df["anomaly_label"].values if "anomaly_label" in df.columns else None
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler, feature_cols
