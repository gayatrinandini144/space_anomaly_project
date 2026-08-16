"""
models/anomaly_detection.py
Phase 2 - Days 8-9: Statistical baseline + ML anomaly detection
Covers: Isolation Forest, One-Class SVM, Statistical Z-score baseline
"""

import numpy as np
import pandas as pd
import joblib
import os
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import (
    classification_report, roc_auc_score,
    precision_recall_curve, confusion_matrix, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


# ── Statistical Baseline (Z-Score) ──────────────────────────────────────────

class StatisticalAnomalyDetector:
    """
    Simple Z-score based anomaly detector.
    Good baseline for comparison (Day 8 - Statistical baseline model).
    """
    def __init__(self, threshold=3.0):
        self.threshold = threshold
        self.means_ = None
        self.stds_ = None

    def fit(self, X):
        self.means_ = np.mean(X, axis=0)
        self.stds_  = np.std(X, axis=0) + 1e-8
        return self

    def predict(self, X):
        z_scores = np.abs((X - self.means_) / self.stds_)
        max_z = z_scores.max(axis=1)
        return (max_z > self.threshold).astype(int)

    def score_samples(self, X):
        z_scores = np.abs((X - self.means_) / self.stds_)
        return -z_scores.max(axis=1)   # negative: higher z = more anomalous


# ── Isolation Forest ─────────────────────────────────────────────────────────

class IsolationForestDetector:
    """
    Isolation Forest anomaly detector (Day 3 & Day 8-9).
    Works well on tabular telemetry without labels.
    """
    def __init__(self, contamination=0.05, n_estimators=100, random_state=42):
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1
        )

    def fit(self, X):
        self.model.fit(X)
        return self

    def predict(self, X):
        raw = self.model.predict(X)
        return ((raw == -1).astype(int))  # -1 → anomaly → 1

    def score_samples(self, X):
        return self.model.score_samples(X)

    def save(self, path="models/isolation_forest.pkl"):
        joblib.dump(self.model, path)
        print(f"✅ Isolation Forest saved to {path}")

    def load(self, path="models/isolation_forest.pkl"):
        self.model = joblib.load(path)
        return self


# ── One-Class SVM ────────────────────────────────────────────────────────────

class OneClassSVMDetector:
    """
    One-Class SVM detector (Day 3).
    Best on smaller datasets; slower on large ones.
    """
    def __init__(self, nu=0.05, kernel="rbf", gamma="scale"):
        self.model = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)

    def fit(self, X):
        self.model.fit(X)
        return self

    def predict(self, X):
        raw = self.model.predict(X)
        return ((raw == -1).astype(int))

    def score_samples(self, X):
        return self.model.score_samples(X)

    def save(self, path="models/one_class_svm.pkl"):
        joblib.dump(self.model, path)

    def load(self, path="models/one_class_svm.pkl"):
        self.model = joblib.load(path)
        return self


# ── Evaluation Utilities ─────────────────────────────────────────────────────

def evaluate_detector(name, y_true, y_pred, scores=None, save_dir="reports"):
    """
    Comprehensive evaluation with ROC-AUC, Precision-Recall, False Alarm Rate.
    Day 12: Model Evaluation.
    """
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'='*50}")
    print(f"  Model: {name}")
    print(f"{'='*50}")
    print(classification_report(y_true, y_pred, target_names=["Normal", "Anomaly"]))

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    print(f"False Alarm Rate : {false_alarm_rate:.4f}")

    if scores is not None:
        roc_auc = roc_auc_score(y_true, -scores)
        print(f"ROC-AUC Score    : {roc_auc:.4f}")

    # ── Plot confusion matrix ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=["Normal", "Anomaly"],
                yticklabels=["Normal", "Anomaly"])
    axes[0].set_title(f"{name} – Confusion Matrix")
    axes[0].set_ylabel("True Label")
    axes[0].set_xlabel("Predicted Label")

    if scores is not None:
        fpr, tpr, _ = roc_curve(y_true, -scores)
        axes[1].plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}", color="steelblue")
        axes[1].plot([0, 1], [0, 1], "k--", alpha=0.4)
        axes[1].set_title(f"{name} – ROC Curve")
        axes[1].set_xlabel("False Positive Rate")
        axes[1].set_ylabel("True Positive Rate")
        axes[1].legend()

    plt.tight_layout()
    fig_path = os.path.join(save_dir, f"{name.replace(' ', '_')}_evaluation.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"📊 Plot saved → {fig_path}")

    return {"roc_auc": roc_auc if scores is not None else None,
            "false_alarm_rate": false_alarm_rate, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


# ── Main Training Pipeline ────────────────────────────────────────────────────

def train_all_detectors(X, y):
    """
    Train all three detectors and compare results.
    """
    # Use only normal samples for unsupervised training
    X_train = X[y == 0]

    print("Training Statistical Baseline...")
    stat_model = StatisticalAnomalyDetector(threshold=3.0)
    stat_model.fit(X_train)
    stat_preds  = stat_model.predict(X)
    stat_scores = stat_model.score_samples(X)

    print("Training Isolation Forest...")
    iso_model = IsolationForestDetector(contamination=0.05)
    iso_model.fit(X_train)
    iso_preds  = iso_model.predict(X)
    iso_scores = iso_model.score_samples(X)
    iso_model.save("models/isolation_forest.pkl")

    print("Training One-Class SVM (may take a minute)...")
    # Subsample for speed
    idx = np.random.choice(len(X_train), min(1000, len(X_train)), replace=False)
    svm_model = OneClassSVMDetector(nu=0.05)
    svm_model.fit(X_train[idx])
    svm_preds  = svm_model.predict(X)
    svm_scores = svm_model.score_samples(X)
    svm_model.save("models/one_class_svm.pkl")

    # Evaluate all
    results = {}
    results["Statistical Z-Score"]  = evaluate_detector("Statistical Z-Score",  y, stat_preds, stat_scores)
    results["Isolation Forest"]      = evaluate_detector("Isolation Forest",      y, iso_preds,  iso_scores)
    results["One-Class SVM"]         = evaluate_detector("One-Class SVM",         y, svm_preds,  svm_scores)

    return stat_model, iso_model, svm_model, results


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    from utils.data_generator import generate_spacecraft_telemetry, add_rul_labels
    from utils.feature_engineering import handle_missing_noisy_data, get_feature_matrix

    df = generate_spacecraft_telemetry(5000)
    df = add_rul_labels(df)
    df = handle_missing_noisy_data(df)
    X, y, scaler, cols = get_feature_matrix(df)

    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    stat_m, iso_m, svm_m, results = train_all_detectors(X, y)
    print("\n✅ All detectors trained and evaluated.")
