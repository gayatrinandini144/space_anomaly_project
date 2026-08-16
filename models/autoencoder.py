"""
models/autoencoder.py
Phase 2 - Day 10: Deep Learning Autoencoder for Anomaly Detection
"""

import numpy as np
import os
import matplotlib.pyplot as plt


def build_autoencoder(input_dim, encoding_dim=16):
    """
    Build a deep autoencoder using TensorFlow/Keras.
    Falls back to a NumPy simulation if TF is not installed.
    """
    try:
        import tensorflow as tf
        from tensorflow import keras

        inputs = keras.Input(shape=(input_dim,))
        # Encoder
        x = keras.layers.Dense(64, activation="relu")(inputs)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.2)(x)
        x = keras.layers.Dense(32, activation="relu")(x)
        encoded = keras.layers.Dense(encoding_dim, activation="relu",
                                     name="latent")(x)
        # Decoder
        x = keras.layers.Dense(32, activation="relu")(encoded)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dense(64, activation="relu")(x)
        decoded = keras.layers.Dense(input_dim, activation="linear")(x)

        autoencoder = keras.Model(inputs, decoded, name="Autoencoder")
        autoencoder.compile(optimizer="adam", loss="mse")
        return autoencoder, "tensorflow"

    except ImportError:
        print("⚠️  TensorFlow not found. Using lightweight NumPy autoencoder simulation.")
        return None, "numpy"


class AutoencoderAnomalyDetector:
    """
    Autoencoder-based anomaly detector.
    Anomalies = high reconstruction error (MSE).
    Day 10: Deep Learning Autoencoder.
    """

    def __init__(self, input_dim, encoding_dim=16, threshold_percentile=95):
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.threshold_percentile = threshold_percentile
        self.threshold_ = None
        self.model, self.backend = build_autoencoder(input_dim, encoding_dim)
        self.history = None

        # NumPy fallback weights
        self._W1 = None
        self._W2 = None

    def _numpy_forward(self, X):
        """Simple linear autoencoder via SVD (numpy fallback)."""
        return X @ self._W1 @ self._W2

    def fit(self, X_train, epochs=50, batch_size=64, validation_split=0.1):
        if self.backend == "tensorflow":
            self.history = self.model.fit(
                X_train, X_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                verbose=1,
                callbacks=[
                    __import__("tensorflow").keras.callbacks.EarlyStopping(
                        patience=5, restore_best_weights=True
                    )
                ]
            )
        else:
            # SVD-based linear autoencoder (CPU friendly)
            U, S, Vt = np.linalg.svd(X_train, full_matrices=False)
            k = self.encoding_dim
            self._W1 = Vt[:k].T          # (input_dim, k)
            self._W2 = Vt[:k]            # (k, input_dim)

        # Compute reconstruction threshold from training data
        recon_errors = self._reconstruction_errors(X_train)
        self.threshold_ = np.percentile(recon_errors, self.threshold_percentile)
        print(f"✅ Autoencoder trained. Anomaly threshold (p{self.threshold_percentile}): {self.threshold_:.6f}")
        return self

    def _reconstruction_errors(self, X):
        if self.backend == "tensorflow":
            X_recon = self.model.predict(X, verbose=0)
        else:
            X_recon = self._numpy_forward(X)
        return np.mean((X - X_recon) ** 2, axis=1)

    def predict(self, X):
        errors = self._reconstruction_errors(X)
        return (errors > self.threshold_).astype(int)

    def score_samples(self, X):
        return -self._reconstruction_errors(X)   # negative: higher error = more anomalous

    def reconstruction_error(self, X):
        return self._reconstruction_errors(X)

    def plot_reconstruction_error(self, X, y_true=None, save_path="reports/autoencoder_error.png"):
        errors = self._reconstruction_errors(X)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        plt.figure(figsize=(14, 5))
        plt.plot(errors, alpha=0.6, color="steelblue", linewidth=0.8, label="Reconstruction Error")
        plt.axhline(self.threshold_, color="red", linestyle="--", linewidth=1.5,
                    label=f"Threshold ({self.threshold_percentile}th pct)")

        if y_true is not None:
            anomaly_idx = np.where(y_true == 1)[0]
            plt.scatter(anomaly_idx, errors[anomaly_idx],
                        color="red", s=20, zorder=5, label="True Anomaly")

        plt.xlabel("Sample Index")
        plt.ylabel("MSE Reconstruction Error")
        plt.title("Autoencoder – Anomaly Detection via Reconstruction Error")
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"📊 Reconstruction error plot saved → {save_path}")

    def plot_training_loss(self, save_path="reports/autoencoder_training_loss.png"):
        if self.history is None:
            print("No training history (numpy backend).")
            return
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.figure(figsize=(8, 4))
        plt.plot(self.history.history["loss"], label="Train Loss")
        plt.plot(self.history.history["val_loss"], label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.title("Autoencoder Training Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"📊 Training loss saved → {save_path}")

    def save(self, path="models/autoencoder"):
        os.makedirs(path, exist_ok=True)
        if self.backend == "tensorflow":
            self.model.save(os.path.join(path, "model.h5"))
        np.save(os.path.join(path, "threshold.npy"), self.threshold_)
        print(f"✅ Autoencoder saved → {path}")

    def load(self, path="models/autoencoder"):
        if self.backend == "tensorflow":
            import tensorflow as tf
            self.model = tf.keras.models.load_model(os.path.join(path, "model.h5"))
        self.threshold_ = np.load(os.path.join(path, "threshold.npy"))
        return self


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    from utils.data_generator import generate_spacecraft_telemetry, add_rul_labels
    from utils.feature_engineering import handle_missing_noisy_data, get_feature_matrix
    from models.anomaly_detection import evaluate_detector

    df = generate_spacecraft_telemetry(5000)
    df = add_rul_labels(df)
    df = handle_missing_noisy_data(df)
    X, y, scaler, cols = get_feature_matrix(df)

    # Train only on normal data
    X_train = X[y == 0]

    ae = AutoencoderAnomalyDetector(input_dim=X.shape[1], encoding_dim=16)
    ae.fit(X_train, epochs=30)
    ae.plot_reconstruction_error(X, y_true=y)
    ae.plot_training_loss()

    preds  = ae.predict(X)
    scores = ae.score_samples(X)
    evaluate_detector("Autoencoder", y, preds, scores)
    ae.save()
