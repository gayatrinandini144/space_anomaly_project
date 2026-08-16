"""
train.py  –  Master Training Pipeline
Runs all 4 phases of the project end-to-end.

Usage:
    python train.py              # train everything
    python train.py --phase 1   # only Phase 1 (foundations / data)
    python train.py --phase 2   # only Phase 2 (ML models)
    python train.py --phase 3   # only Phase 3 (decision layer)
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import joblib

# Allow imports from project root
sys.path.insert(0, os.path.dirname(__file__))

from utils.data_generator      import generate_spacecraft_telemetry, add_rul_labels
from utils.feature_engineering import (
    handle_missing_noisy_data, get_feature_matrix, add_subsystem_health_scores
)
from models.anomaly_detection  import train_all_detectors
from models.autoencoder        import AutoencoderAnomalyDetector
from models.lstm_model         import LSTMAnomalyDetector, create_sequences
from models.decision_engine    import simulate_mission

os.makedirs("data",    exist_ok=True)
os.makedirs("models",  exist_ok=True)
os.makedirs("reports", exist_ok=True)


def phase1_data():
    """Day 1-5: Generate and explore telemetry data."""
    print("\n" + "="*60)
    print("  PHASE 1: Foundations – Data Generation & Exploration")
    print("="*60)

    df = generate_spacecraft_telemetry(n_samples=5000, anomaly_fraction=0.05)
    df = add_rul_labels(df)
    df = handle_missing_noisy_data(df)
    df = add_subsystem_health_scores(df)
    df.to_csv("data/spacecraft_telemetry.csv", index=False)

    print(f"\n📡 Dataset Shape   : {df.shape}")
    print(f"🔴 Anomaly Count   : {df['anomaly_label'].sum()} ({df['anomaly_label'].mean():.1%})")
    print(f"📊 Subsystems      : Power, Thermal, Propulsion, Communication")
    print(f"🕐 Duration        : {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"💾 Saved to        : data/spacecraft_telemetry.csv\n")
    return df


def phase2_models(df):
    """Day 6-12: Build and evaluate all anomaly detection models."""
    print("\n" + "="*60)
    print("  PHASE 2: Model Development")
    print("="*60)

    X, y, scaler, cols = get_feature_matrix(df)
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(cols,   "models/feature_cols.pkl")

    # ── 2a. Statistical + ML models (Days 8-9) ──────────────────────────────
    print("\n[2a] Statistical Baseline + ML Detectors (Day 8-9)")
    stat_m, iso_m, svm_m, results = train_all_detectors(X, y)
    joblib.dump(results, "reports/ml_results.pkl")

    # ── 2b. Deep Learning Autoencoder (Day 10) ───────────────────────────────
    print("\n[2b] Autoencoder (Day 10)")
    X_train = X[y == 0]
    ae = AutoencoderAnomalyDetector(input_dim=X.shape[1], encoding_dim=16)
    ae.fit(X_train, epochs=30)
    ae.plot_reconstruction_error(X, y_true=y)
    ae.save("models/autoencoder")

    # ── 2c. LSTM for RUL prediction (Days 4, 11) ─────────────────────────────
    print("\n[2c] LSTM RUL Predictor (Day 11)")
    y_rul = df["rul"].values
    SEQ_LEN = 30
    X_seq, y_seq = create_sequences(X, y_rul, seq_len=SEQ_LEN)
    y_norm = y_seq / (y_seq.max() + 1e-8)

    lstm = LSTMAnomalyDetector(seq_len=SEQ_LEN, n_features=X.shape[1])
    lstm.fit(X_seq, y_norm, epochs=20)
    lstm.plot_rul_prediction(X_seq, y_seq)
    lstm.save("models/lstm")

    print("\n✅ Phase 2 complete. Models saved to models/")
    return iso_m, ae, lstm, scaler


def phase3_decisions(df):
    """Day 13-17: Autonomous decision layer."""
    print("\n" + "="*60)
    print("  PHASE 3: Autonomous Decision Layer")
    print("="*60)

    summary = simulate_mission(df, n_steps=200)

    print("\nDecision Summary:")
    print(summary["severity"].value_counts().to_string())
    print("\nRecommended Actions:")
    print(summary["recommended_action"].value_counts().to_string())

    summary.to_csv("reports/decision_summary.csv", index=False)
    print("\n✅ Phase 3 complete. Decision log → reports/decision_log.json")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Space Anomaly Detection – Training Pipeline")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3, 4], default=0,
                        help="Run a specific phase (1-3). Default: run all.")
    args = parser.parse_args()

    if args.phase == 0 or args.phase == 1:
        df = phase1_data()
    else:
        df = pd.read_csv("data/spacecraft_telemetry.csv", parse_dates=["timestamp"])
        df = add_subsystem_health_scores(df)

    if args.phase == 0 or args.phase == 2:
        phase2_models(df)

    if args.phase == 0 or args.phase == 3:
        phase3_decisions(df)

    print("\n" + "="*60)
    print("  🚀 All phases complete!")
    print("  Run the dashboard: streamlit run dashboard/app.py")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
