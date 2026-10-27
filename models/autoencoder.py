"""
models/autoencoder.py — SVD Autoencoder (no TensorFlow needed)
"""
import numpy as np, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt


class AutoencoderAnomalyDetector:
    def __init__(self, encoding_dim=16, threshold_percentile=95):
        self.encoding_dim         = encoding_dim
        self.threshold_percentile = threshold_percentile
        self.threshold_           = None
        self._W_encode = None
        self._W_decode = None

    def fit(self, X_train):
        U,S,Vt = np.linalg.svd(X_train, full_matrices=False)
        k = min(self.encoding_dim, Vt.shape[0])
        self._W_encode = Vt[:k].T
        self._W_decode = Vt[:k]
        errs = self._errors(X_train)
        self.threshold_ = np.percentile(errs, self.threshold_percentile)
        print(f"  Autoencoder trained | threshold: {self.threshold_:.6f}")
        return self

    def _errors(self, X):
        recon = (X @ self._W_encode) @ self._W_decode
        return np.mean((X-recon)**2, axis=1)

    def predict(self, X):
        return (self._errors(X) > self.threshold_).astype(int)

    def score_samples(self, X):
        return -self._errors(X)

    def plot_reconstruction_error(self, X, y_true=None,
                                   save_path="reports/autoencoder_error.png"):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        errs = self._errors(X)
        plt.figure(figsize=(14,4))
        plt.plot(errs, alpha=0.6, color="steelblue", lw=0.8, label="Recon Error")
        plt.axhline(self.threshold_, color="red", ls="--",
                    label=f"Threshold (p{self.threshold_percentile})")
        if y_true is not None:
            idx = np.where(y_true==1)[0]
            plt.scatter(idx, errs[idx], color="red", s=15, zorder=5, label="Anomaly")
        plt.xlabel("Sample"); plt.ylabel("MSE")
        plt.title("Autoencoder – Reconstruction Error"); plt.legend()
        plt.tight_layout(); plt.savefig(save_path, dpi=120); plt.close()

    def save(self, d="models/autoencoder"):
        os.makedirs(d, exist_ok=True)
        np.save(f"{d}/W_encode.npy",  self._W_encode)
        np.save(f"{d}/W_decode.npy",  self._W_decode)
        np.save(f"{d}/threshold.npy", self.threshold_)

    def load(self, d="models/autoencoder"):
        self._W_encode  = np.load(f"{d}/W_encode.npy")
        self._W_decode  = np.load(f"{d}/W_decode.npy")
        self.threshold_ = float(np.load(f"{d}/threshold.npy"))
        return self
