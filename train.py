"""
train.py — Master Training Pipeline
Usage:
    python train.py              # run all phases
    python train.py --phase 1   # data generation
    python train.py --phase 2   # ML models
    python train.py --phase 3   # decision engine
"""
import os, sys, argparse
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(__file__))

from utils.data_generator      import (generate_spacecraft_telemetry,
                                        add_rul_labels, add_subsystem_health_scores,
                                        load_nasa_cmapss)
from utils.feature_engineering import (handle_missing_noisy_data, get_feature_matrix)
from models.anomaly_detection  import train_all_detectors
from models.autoencoder        import AutoencoderAnomalyDetector
from models.lstm_model         import LSTMAnomalyDetector, create_sequences
from models.decision_engine    import simulate_mission

os.makedirs("data",    exist_ok=True)
os.makedirs("models",  exist_ok=True)
os.makedirs("reports", exist_ok=True)


def phase1():
    print("\n"+"="*55)
    print("  PHASE 1 — Data Generation  (Day 1-5)")
    print("="*55)
    df = generate_spacecraft_telemetry(5000, anomaly_fraction=0.05)
    df = add_rul_labels(df)
    df = handle_missing_noisy_data(df)
    df = add_subsystem_health_scores(df)
    df.to_csv("data/spacecraft_telemetry.csv", index=False)
    print(f"\n  Rows      : {len(df)}")
    print(f"  Anomalies : {df['anomaly_label'].sum()} ({df['anomaly_label'].mean():.1%})")
    print(f"  Saved     : data/spacecraft_telemetry.csv")

    # Try NASA data too
    nasa = load_nasa_cmapss("data/nasa_cmapss")
    if nasa is not None:
        nasa.to_csv("data/nasa_cmapss_processed.csv", index=False)
        print(f"  NASA CMAPSS saved: data/nasa_cmapss_processed.csv")
    return df


def phase2(df):
    print("\n"+"="*55)
    print("  PHASE 2 — Model Development  (Day 6-12)")
    print("="*55)
    X, y, scaler, cols = get_feature_matrix(df)
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(cols,   "models/feature_cols.pkl")
    print(f"\n  Features: {X.shape[1]} | Samples: {X.shape[0]} | Anomaly rate: {y.mean():.2%}")

    print("\n[2a] ML Detectors (Day 8-9)")
    _, _, _, results = train_all_detectors(X, y)
    joblib.dump(results, "reports/ml_results.pkl")

    print("\n[2b] Autoencoder (Day 10)")
    ae = AutoencoderAnomalyDetector(encoding_dim=16)
    ae.fit(X[y==0])
    ae.plot_reconstruction_error(X, y_true=y)
    ae.save("models/autoencoder")

    print("\n[2c] LSTM + RUL (Day 11)")
    y_rul = df["rul"].values
    X_seq, y_seq = create_sequences(X, y_rul, seq_len=30)
    y_norm = y_seq / (y_seq.max()+1e-8)
    lstm = LSTMAnomalyDetector(seq_len=30, n_features=X.shape[1])
    lstm.fit(X_seq, y_norm)
    lstm.evaluate_rul(X_seq, y_norm)
    lstm.plot_rul_prediction(X_seq, y_seq)
    lstm.save("models/lstm")
    print("\n  ✅ Phase 2 complete")


def phase3(df):
    print("\n"+"="*55)
    print("  PHASE 3 — Autonomous Decisions  (Day 13-17)")
    print("="*55)
    summary = simulate_mission(df, n_steps=300)
    print("\n  Severity:"); print(summary["severity"].value_counts().to_string())
    print("\n  Actions:");  print(summary["recommended_action"].value_counts().to_string())
    summary.to_csv("reports/decision_summary.csv", index=False)
    print("\n  ✅ Phase 3 complete")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, choices=[1,2,3], default=0)
    args = parser.parse_args()

    if args.phase in (0,1):
        df = phase1()
    else:
        print("  Loading existing dataset...")
        df = pd.read_csv("data/spacecraft_telemetry.csv", parse_dates=["timestamp"])
        df = add_subsystem_health_scores(df)

    if args.phase in (0,2): phase2(df)
    if args.phase in (0,3): phase3(df)

    print("\n"+"="*55)
    print("  🚀 Complete! Now run:")
    print("     streamlit run dashboard/app.py")
    print("="*55+"\n")

if __name__=="__main__":
    main()
