"""
utils/feature_engineering.py
Time-series feature engineering for spacecraft telemetry (Day 2 & Day 6-7)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


SENSOR_COLS = [
    "solar_panel_voltage", "battery_charge", "power_consumption",
    "internal_temp", "external_temp", "thermal_control_power",
    "fuel_pressure", "thruster_temp", "delta_v",
    "signal_strength", "data_rate", "bit_error_rate",
]


def add_rolling_features(df, windows=(5, 10, 20)):
    """
    Add rolling mean, std, min, max features for anomaly context.
    Critical for LSTM and statistical baseline models.
    """
    df = df.copy()
    for col in SENSOR_COLS:
        for w in windows:
            df[f"{col}_roll_mean_{w}"] = df[col].rolling(w, min_periods=1).mean()
            df[f"{col}_roll_std_{w}"]  = df[col].rolling(w, min_periods=1).std().fillna(0)
    return df


def add_diff_features(df):
    """
    First-order difference (rate of change) - catches sudden spikes.
    """
    df = df.copy()
    for col in SENSOR_COLS:
        df[f"{col}_diff1"] = df[col].diff().fillna(0)
    return df


def add_subsystem_health_scores(df):
    """
    Composite health score per subsystem (0=healthy, 1=critical).
    Used in the risk scoring layer (Day 16).
    """
    df = df.copy()

    # Power: low battery + high consumption = bad
    v_norm = (df["solar_panel_voltage"] - 28.5) / 0.5
    b_norm = (df["battery_charge"] - 60) / 20
    p_norm = (df["power_consumption"] - 450) / 20
    df["power_health"] = np.clip(1 - (np.abs(v_norm) + np.abs(p_norm) - b_norm) / 3, 0, 1)

    # Thermal: temperature deviation
    t_norm = (df["internal_temp"] - 22) / 2
    df["thermal_health"] = np.clip(1 - np.abs(t_norm) / 5, 0, 1)

    # Propulsion: fuel pressure drop
    f_norm = (df["fuel_pressure"] - 150) / 3
    df["propulsion_health"] = np.clip(1 - np.abs(f_norm) / 10, 0, 1)

    # Communication: bit error rate spike
    df["comm_health"] = np.clip(1 - df["bit_error_rate"] / 0.05, 0, 1)

    # Overall risk score
    df["mission_risk_score"] = 1 - (
        0.3 * df["power_health"]
        + 0.25 * df["thermal_health"]
        + 0.25 * df["propulsion_health"]
        + 0.2 * df["comm_health"]
    )

    return df


def handle_missing_noisy_data(df):
    """
    Handle missing values and noise in space telemetry (Day 6-7).
    """
    df = df.copy()

    # Forward-fill short gaps (common in telemetry dropouts)
    df[SENSOR_COLS] = df[SENSOR_COLS].ffill(limit=3)

    # Clip extreme outliers (>5 sigma) - could be sensor errors
    for col in SENSOR_COLS:
        mean, std = df[col].mean(), df[col].std()
        df[col] = df[col].clip(mean - 5 * std, mean + 5 * std)

    return df


def get_feature_matrix(df, include_rolling=True, include_diff=True):
    """
    Build final feature matrix for model training.
    """
    if include_rolling:
        df = add_rolling_features(df)
    if include_diff:
        df = add_diff_features(df)

    df = add_subsystem_health_scores(df)

    feature_cols = [c for c in df.columns if c not in
                    ["timestamp", "anomaly_label", "rul"]]
    X = df[feature_cols].values
    y = df["anomaly_label"].values if "anomaly_label" in df.columns else None

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler, feature_cols


if __name__ == "__main__":
    from data_generator import generate_spacecraft_telemetry, add_rul_labels
    df = generate_spacecraft_telemetry()
    df = add_rul_labels(df)
    df = handle_missing_noisy_data(df)
    X, y, scaler, cols = get_feature_matrix(df)
    print(f"✅ Feature matrix shape: {X.shape}  | Anomaly rate: {y.mean():.2%}")
