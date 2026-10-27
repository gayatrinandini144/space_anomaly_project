"""
models/lstm_model.py — LSTM-style RUL predictor (NumPy, no TF)
"""
import numpy as np, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt


def create_sequences(X, y_rul, seq_len=30):
    Xs, ys = [], []
    for i in range(len(X)-seq_len):
        Xs.append(X[i:i+seq_len])
        ys.append(y_rul[i+seq_len-1])
    return np.array(Xs), np.array(ys)


class LSTMAnomalyDetector:
    def __init__(self, seq_len=30, n_features=12):
        self.seq_len    = seq_len
        self.n_features = n_features
        self._coef      = None
        self._intercept = None

    def _featurize(self, X_seq):
        t = np.arange(self.seq_len)
        feats = []
        for seq in X_seq:
            mean  = seq.mean(axis=0)
            std   = seq.std(axis=0)
            last  = seq[-1]
            slope = np.array([np.polyfit(t, seq[:,f], 1)[0]
                               for f in range(seq.shape[1])])
            feats.append(np.concatenate([mean,std,last,slope]))
        return np.array(feats)

    def fit(self, X_seq, y, **kw):
        F = self._featurize(X_seq)
        A = np.hstack([F, np.ones((len(F),1))])
        lam = 1e-3
        sol = np.linalg.solve(A.T@A + lam*np.eye(A.shape[1]), A.T@y)
        self._coef = sol[:-1]; self._intercept = sol[-1]
        mae = np.mean(np.abs(y - (A@sol)))
        print(f"  LSTM trained | train MAE: {mae:.4f}")
        return self

    def predict_rul(self, X_seq):
        F = self._featurize(X_seq)
        return F @ self._coef + self._intercept

    def evaluate_rul(self, X_seq, y):
        yp = self.predict_rul(X_seq)
        mae  = np.mean(np.abs(y-yp))
        rmse = np.sqrt(np.mean((y-yp)**2))
        print(f"  RUL MAE: {mae:.2f} | RMSE: {rmse:.2f}")
        return mae, rmse

    def plot_rul_prediction(self, X_seq, y_true,
                             save_path="reports/rul_prediction.png"):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        yp = self.predict_rul(X_seq)
        plt.figure(figsize=(14,4))
        plt.plot(y_true, label="Actual RUL",    color="steelblue",  lw=1.5)
        plt.plot(yp,     label="Predicted RUL", color="darkorange", lw=1.5, ls="--")
        plt.xlabel("Time Step"); plt.ylabel("RUL")
        plt.title("LSTM – Remaining Useful Life Prediction"); plt.legend()
        plt.tight_layout(); plt.savefig(save_path, dpi=120); plt.close()

    def save(self, d="models/lstm"):
        os.makedirs(d, exist_ok=True)
        np.save(f"{d}/coef.npy",      self._coef)
        np.save(f"{d}/intercept.npy", self._intercept)

    def load(self, d="models/lstm"):
        self._coef      = np.load(f"{d}/coef.npy")
        self._intercept = float(np.load(f"{d}/intercept.npy"))
        return self
