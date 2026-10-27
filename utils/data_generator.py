"""
utils/data_generator.py
Spacecraft telemetry simulation + NASA CMAPSS integration
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os


SENSOR_COLS = [
    "solar_panel_voltage", "battery_charge", "power_consumption",
    "internal_temp", "external_temp", "thermal_control_power",
    "fuel_pressure", "thruster_temp", "delta_v",
    "signal_strength", "data_rate", "bit_error_rate",
]


def generate_spacecraft_telemetry(n_samples=5000, anomaly_fraction=0.05, seed=42):
    np.random.seed(seed)
    start = datetime(2024, 1, 1)
    timestamps = [start + timedelta(minutes=i * 5) for i in range(n_samples)]

    solar_panel_voltage   = np.random.normal(28.5, 0.5, n_samples)
    battery_charge        = np.clip(np.cumsum(np.random.normal(0, 0.1, n_samples)) % 100 + 60, 20, 100)
    power_consumption     = np.random.normal(450, 20, n_samples)
    internal_temp         = np.random.normal(22, 2, n_samples)
    external_temp         = np.random.normal(-50, 10, n_samples)
    thermal_control_power = np.random.normal(80, 5, n_samples)
    fuel_pressure         = np.random.normal(150, 3, n_samples)
    thruster_temp         = np.random.normal(200, 10, n_samples)
    delta_v               = np.random.normal(0.5, 0.1, n_samples)
    signal_strength       = np.random.normal(-80, 5, n_samples)
    data_rate             = np.random.normal(100, 10, n_samples)
    bit_error_rate        = np.random.exponential(0.001, n_samples)

    n_anom = int(n_samples * anomaly_fraction)
    anom_idx = np.random.choice(n_samples, n_anom, replace=False)
    labels = np.zeros(n_samples, dtype=int)
    labels[anom_idx] = 1

    anomaly_types = []
    type_arr = ["normal"] * n_samples
    for idx in anom_idx:
        t = np.random.choice(["power_spike","thermal_runaway","fuel_leak","comm_dropout"],
                             p=[0.3,0.3,0.2,0.2])
        type_arr[idx] = t
        if t == "power_spike":
            solar_panel_voltage[idx] *= np.random.uniform(1.3,1.8)
            power_consumption[idx]   *= np.random.uniform(1.5,2.5)
            battery_charge[idx]       = np.random.uniform(5,15)
        elif t == "thermal_runaway":
            internal_temp[idx]         += np.random.uniform(15,40)
            thermal_control_power[idx] *= np.random.uniform(2,4)
            thruster_temp[idx]         += np.random.uniform(50,100)
        elif t == "fuel_leak":
            fuel_pressure[idx] *= np.random.uniform(0.4,0.7)
            delta_v[idx]       *= np.random.uniform(2,5)
        elif t == "comm_dropout":
            signal_strength[idx] += np.random.uniform(20,40)
            data_rate[idx]       *= np.random.uniform(0.1,0.4)
            bit_error_rate[idx]  *= np.random.uniform(50,200)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "solar_panel_voltage": solar_panel_voltage,
        "battery_charge": battery_charge,
        "power_consumption": power_consumption,
        "internal_temp": internal_temp,
        "external_temp": external_temp,
        "thermal_control_power": thermal_control_power,
        "fuel_pressure": fuel_pressure,
        "thruster_temp": thruster_temp,
        "delta_v": delta_v,
        "signal_strength": signal_strength,
        "data_rate": data_rate,
        "bit_error_rate": bit_error_rate,
        "anomaly_label": labels,
        "anomaly_type": type_arr,
    })
    return df


def add_rul_labels(df):
    df = df.copy()
    df["rul"] = list(range(len(df)-1, -1, -1))
    return df


def add_subsystem_health_scores(df):
    df = df.copy()
    v = (df["solar_panel_voltage"]-28.5)/0.5
    b = (df["battery_charge"]-60)/20
    p = (df["power_consumption"]-450)/20
    df["power_health"]      = np.clip(1-(np.abs(v)+np.abs(p)-b)/3, 0, 1)
    t = (df["internal_temp"]-22)/2
    df["thermal_health"]    = np.clip(1-np.abs(t)/5, 0, 1)
    f = (df["fuel_pressure"]-150)/3
    df["propulsion_health"] = np.clip(1-np.abs(f)/10, 0, 1)
    df["comm_health"]       = np.clip(1-df["bit_error_rate"]/0.05, 0, 1)
    df["mission_risk_score"]= 1-(
        0.30*df["power_health"]+0.25*df["thermal_health"]
       +0.25*df["propulsion_health"]+0.20*df["comm_health"]
    )
    return df


def load_nasa_cmapss(data_dir="data/nasa_cmapss"):
    """
    Load NASA CMAPSS dataset if available.
    Download from: https://www.kaggle.com/datasets/behrad3d/nasa-cmaps
    Place train_FD001.txt in data/nasa_cmapss/
    """
    path = os.path.join(data_dir, "train_FD001.txt")
    if not os.path.exists(path):
        print(f"NASA CMAPSS not found at {path}. Using simulated data.")
        return None

    cols = ["unit","cycle","op1","op2","op3"] + [f"s{i}" for i in range(1,22)]
    df = pd.read_csv(path, sep=r"\s+", header=None, names=cols)

    # Compute RUL
    max_cycles = df.groupby("unit")["cycle"].max().reset_index()
    max_cycles.columns = ["unit","max_cycle"]
    df = df.merge(max_cycles, on="unit")
    df["rul"] = df["max_cycle"] - df["cycle"]
    df["anomaly_label"] = (df["rul"] < 30).astype(int)
    df.drop("max_cycle", axis=1, inplace=True)
    print(f"NASA CMAPSS loaded: {len(df)} rows | engines: {df['unit'].nunique()}")
    return df
