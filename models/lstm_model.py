"""
models/lstm_model.py
Phase 1 - Day 4 & Phase 2 - Day 11:
  - LSTM for Predictive Failure Detection
  - Predict Remaining Useful Life (RUL)
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


SENSOR_COLS = [
    "solar_panel_voltage", "battery_charge", "power_consumption",
    "internal_temp", "external_temp", "thermal_control_power",
    "fuel_pressure", "thruster_temp", "delta_v",
    "signal_strength", "data_rate", "bit_error_rate",
]


def create_sequences(X, y_rul, seq_len=30):
    """
    Slide a window of length `seq_len` over the telemetry stream.
    Returns (sequences, rul_targets) for LSTM training.
    """
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i : i + seq_len])
        ys.append(y_rul[i + seq_len - 1])
    return np.array(Xs), np.array(ys)


def build_lstm_model(seq_len, n_features):
    """
    Stacked LSTM for RUL regression.
    Falls back to a simple linear extrapolation if TF not available.
    """
    try:
        import tensorflow as tf
        from tensorflow import keras

        model = keras.Sequential([
            keras.layers.LSTM(64, return_sequences=True,
                              input_shape=(seq_len, n_features)),
            keras.layers.Dropout(0.2),
            keras.layers.LSTM(32),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(1, activation="linear", name="rul_output"),
        ], name="LSTM_RUL")

        model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        return model, "tensorflow"

    except ImportError:
        return None, "numpy"


class LSTMAnomalyDetector:
    """
    LSTM-based model for:
    1. Anomaly detection on sliding windows
    2. RUL (Remaining Useful Life) prediction
    """

    def __init__(self, seq_len=30, n_features=12, mode="rul"):
        self.seq_len = seq_len
        self.n_features = n_features
        self.mode = mode   # "rul" or "classification"
        self.model, self.backend = build_lstm_model(seq_len, n_features)
        self.history = None
        self._linear_coef = None

    def _numpy_fit(self, X_seq, y):
        """Fallback: simple linear regression on mean sequence values."""
        X_flat = X_seq.mean(axis=1).mean(axis=1, keepdims=True)
        A = np.hstack([X_flat, np.ones((len(X_flat), 1))])
        self._linear_coef = np.linalg.lstsq(A, y, rcond=None)[0]

    def _numpy_predict(self, X_seq):
        X_flat = X_seq.mean(axis=1).mean(axis=1, keepdims=True)
        A = np.hstack([X_flat, np.ones((len(X_flat), 1))])
        return A @ self._linear_coef

    def fit(self, X_seq, y, epochs=30, batch_size=64, validation_split=0.1):
        if self.backend == "tensorflow":
            callbacks = [
                __import__("tensorflow").keras.callbacks.EarlyStopping(
                    patience=5, restore_best_weights=True
                ),
                __import__("tensorflow").keras.callbacks.ReduceLROnPlateau(
                    factor=0.5, patience=3
                ),
            ]
            self.history = self.model.fit(
                X_seq, y,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                verbose=1,
                callbacks=callbacks,
            )
        else:
            self._numpy_fit(X_seq, y)
            print("✅ LSTM (numpy fallback) trained via linear regression.")

        return self

    def predict_rul(self, X_seq):
        if self.backend == "tensorflow":
            return self.model.predict(X_seq, verbose=0).flatten()
        return self._numpy_predict(X_seq)

    def evaluate_rul(self, X_seq, y_true):
        y_pred = self.predict_rul(X_seq)
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        print(f"RUL MAE : {mae:.2f} | RMSE : {rmse:.2f}")
        return mae, rmse

    def plot_rul_prediction(self, X_seq, y_true, save_path="reports/rul_prediction.png"):
        y_pred = self.predict_rul(X_seq)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        plt.figure(figsize=(14, 5))
        plt.plot(y_true, label="Actual RUL", color="steelblue", linewidth=1.5)
        plt.plot(y_pred, label="Predicted RUL", color="darkorange",
                 linewidth=1.5, linestyle="--")
        plt.xlabel("Time Step")
        plt.ylabel("Remaining Useful Life (cycles)")
        plt.title("LSTM – Remaining Useful Life Prediction")
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"📊 RUL plot saved → {save_path}")

    def plot_training_loss(self, save_path="reports/lstm_training_loss.png"):
        if self.history is None:
            return
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.figure(figsize=(8, 4))
        plt.plot(self.history.history["loss"], label="Train Loss")
        plt.plot(self.history.history["val_loss"], label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.title("LSTM Training Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()

    def save(self, path="models/lstm"):
        os.makedirs(path, exist_ok=True)
        if self.backend == "tensorflow":
            self.model.save(os.path.join(path, "model.h5"))
            print(f"✅ LSTM saved → {path}")

    def load(self, path="models/lstm"):
        if self.backend == "tensorflow":
            import tensorflow as tf
            self.model = tf.keras.models.load_model(os.path.join(path, "model.h5"))
        return self


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    from utils.data_generator import generate_spacecraft_telemetry, add_rul_labels
    from utils.feature_engineering import handle_missing_noisy_data, get_feature_matrix
    from sklearn.preprocessing import MinMaxScaler

    df = generate_spacecraft_telemetry(5000)
    df = add_rul_labels(df)
    df = handle_missing_noisy_data(df)

    X, y, scaler, cols = get_feature_matrix(df)
    y_rul = df["rul"].values[len(df) - len(X):]   # align

    SEQ_LEN = 30
    X_seq, y_seq = create_sequences(X, y_rul, seq_len=SEQ_LEN)

    # Normalize RUL
    rul_max = y_seq.max()
    y_norm = y_seq / rul_max

    lstm = LSTMAnomalyDetector(seq_len=SEQ_LEN, n_features=X.shape[1])
    lstm.fit(X_seq, y_norm, epochs=20)
    lstm.plot_rul_prediction(X_seq, y_seq)
    lstm.plot_training_loss()
    mae, rmse = lstm.evaluate_rul(X_seq, y_norm)
    lstm.save()
