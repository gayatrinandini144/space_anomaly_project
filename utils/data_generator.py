"""
utils/data_generator.py
Simulates spacecraft subsystem telemetry data (Phase 1 - Day 1/2)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_spacecraft_telemetry(n_samples=5000, anomaly_fraction=0.05, seed=42):
    """
    Generate realistic spacecraft telemetry data with injected anomalies.
    Simulates sensors across 4 subsystems: Power, Thermal, Propulsion, Communication.
    
    Returns:
        pd.DataFrame with timestamp, sensor readings, and anomaly labels
    """
    np.random.seed(seed)
    
    start_time = datetime(2024, 1, 1, 0, 0, 0)
    timestamps = [start_time + timedelta(minutes=i * 5) for i in range(n_samples)]

    # ── Power Subsystem ──────────────────────────────────────────────────────
    solar_panel_voltage   = np.random.normal(28.5, 0.5, n_samples)   # Volts
    battery_charge        = np.clip(np.cumsum(np.random.normal(0, 0.1, n_samples)) % 100 + 60, 20, 100)
    power_consumption     = np.random.normal(450, 20, n_samples)      # Watts

    # ── Thermal Subsystem ────────────────────────────────────────────────────
    internal_temp         = np.random.normal(22, 2, n_samples)        # Celsius
    external_temp         = np.random.normal(-50, 10, n_samples)      # Celsius (deep space)
    thermal_control_power = np.random.normal(80, 5, n_samples)        # Watts

    # ── Propulsion Subsystem ─────────────────────────────────────────────────
    fuel_pressure         = np.random.normal(150, 3, n_samples)       # PSI
    thruster_temp         = np.random.normal(200, 10, n_samples)      # Celsius
    delta_v               = np.random.normal(0.5, 0.1, n_samples)     # m/s

    # ── Communication Subsystem ──────────────────────────────────────────────
    signal_strength       = np.random.normal(-80, 5, n_samples)       # dBm
    data_rate             = np.random.normal(100, 10, n_samples)      # Mbps
    bit_error_rate        = np.random.exponential(0.001, n_samples)   # Ratio

    # ── Inject Anomalies ─────────────────────────────────────────────────────
    n_anomalies = int(n_samples * anomaly_fraction)
    anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)
    labels = np.zeros(n_samples, dtype=int)
    labels[anomaly_indices] = 1

    for idx in anomaly_indices:
        anomaly_type = np.random.choice(
            ["power_spike", "thermal_runaway", "fuel_leak", "comm_dropout"], p=[0.3, 0.3, 0.2, 0.2]
        )
        if anomaly_type == "power_spike":
            solar_panel_voltage[idx]   *= np.random.uniform(1.3, 1.8)
            power_consumption[idx]     *= np.random.uniform(1.5, 2.5)
            battery_charge[idx]        = np.random.uniform(5, 15)
        elif anomaly_type == "thermal_runaway":
            internal_temp[idx]         += np.random.uniform(15, 40)
            thermal_control_power[idx] *= np.random.uniform(2, 4)
            thruster_temp[idx]         += np.random.uniform(50, 100)
        elif anomaly_type == "fuel_leak":
            fuel_pressure[idx]         *= np.random.uniform(0.4, 0.7)
            delta_v[idx]               *= np.random.uniform(2, 5)
        elif anomaly_type == "comm_dropout":
            signal_strength[idx]       += np.random.uniform(20, 40)   # weaker (less negative)
            data_rate[idx]             *= np.random.uniform(0.1, 0.4)
            bit_error_rate[idx]        *= np.random.uniform(50, 200)

    df = pd.DataFrame({
        "timestamp":            timestamps,
        "solar_panel_voltage":  solar_panel_voltage,
        "battery_charge":       battery_charge,
        "power_consumption":    power_consumption,
        "internal_temp":        internal_temp,
        "external_temp":        external_temp,
        "thermal_control_power":thermal_control_power,
        "fuel_pressure":        fuel_pressure,
        "thruster_temp":        thruster_temp,
        "delta_v":              delta_v,
        "signal_strength":      signal_strength,
        "data_rate":            data_rate,
        "bit_error_rate":       bit_error_rate,
        "anomaly_label":        labels,
    })

    return df


def add_rul_labels(df):
    """
    Compute Remaining Useful Life (RUL) from battery_charge degradation.
    Used in Day 11 (Predictive Failure Detection).
    """
    rul = []
    last_idx = len(df) - 1
    for i in range(len(df)):
        remaining = last_idx - i
        rul.append(remaining)
    df["rul"] = rul
    return df


if __name__ == "__main__":
    df = generate_spacecraft_telemetry(n_samples=5000, anomaly_fraction=0.05)
    df = add_rul_labels(df)
    df.to_csv("data/spacecraft_telemetry.csv", index=False)
    print(f"✅ Dataset saved: {len(df)} rows | Anomalies: {df['anomaly_label'].sum()}")
    print(df.head())
