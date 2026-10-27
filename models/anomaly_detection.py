"""
models/anomaly_detection.py
Statistical Z-Score + Isolation Forest + One-Class SVM
"""
import numpy as np
import joblib, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import (classification_report, roc_auc_score,
                             confusion_matrix, roc_curve, precision_recall_curve)


class StatisticalAnomalyDetector:
    def __init__(self, threshold=3.0):
        self.threshold = threshold
    def fit(self, X):
        self.means_ = np.mean(X, axis=0)
        self.stds_  = np.std(X, axis=0) + 1e-8
        return self
    def predict(self, X):
        z = np.abs((X-self.means_)/self.stds_).max(axis=1)
        return (z > self.threshold).astype(int)
    def score_samples(self, X):
        return -np.abs((X-self.means_)/self.stds_).max(axis=1)


class IsolationForestDetector:
    def __init__(self, contamination=0.05):
        self.model = IsolationForest(contamination=contamination,
                                     n_estimators=100, random_state=42, n_jobs=-1)
    def fit(self, X):   self.model.fit(X); return self
    def predict(self, X): return (self.model.predict(X)==-1).astype(int)
    def score_samples(self, X): return self.model.score_samples(X)
    def save(self, p): joblib.dump(self.model, p)


class OneClassSVMDetector:
    def __init__(self, nu=0.05):
        self.model = OneClassSVM(nu=nu, kernel="rbf", gamma="scale")
    def fit(self, X):   self.model.fit(X); return self
    def predict(self, X): return (self.model.predict(X)==-1).astype(int)
    def score_samples(self, X): return self.model.score_samples(X)
    def save(self, p): joblib.dump(self.model, p)


def evaluate_detector(name, y_true, y_pred, scores=None, save_dir="reports"):
    os.makedirs(save_dir, exist_ok=True)
    print(f"\n{'='*50}\n  {name}\n{'='*50}")
    print(classification_report(y_true, y_pred, target_names=["Normal","Anomaly"]))
    cm = confusion_matrix(y_true, y_pred)
    tn,fp,fn,tp = cm.ravel()
    far = fp/(fp+tn+1e-8)
    roc_auc = None
    if scores is not None:
        try: roc_auc = roc_auc_score(y_true, -scores)
        except: pass

    fig, axes = plt.subplots(1,3,figsize=(15,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=["Normal","Anomaly"], yticklabels=["Normal","Anomaly"])
    axes[0].set_title(f"{name}\nConfusion Matrix")
    axes[0].set_ylabel("True"); axes[0].set_xlabel("Predicted")

    if scores is not None and roc_auc:
        fpr,tpr,_ = roc_curve(y_true,-scores)
        axes[1].plot(fpr,tpr,label=f"AUC={roc_auc:.3f}",color="steelblue",lw=2)
        axes[1].plot([0,1],[0,1],"k--",alpha=0.4)
        axes[1].set_title("ROC Curve"); axes[1].set_xlabel("FPR"); axes[1].set_ylabel("TPR")
        axes[1].legend()

        prec,rec,_ = precision_recall_curve(y_true,-scores)
        axes[2].plot(rec,prec,color="darkorange",lw=2)
        axes[2].set_title("Precision-Recall Curve")
        axes[2].set_xlabel("Recall"); axes[2].set_ylabel("Precision")

    plt.suptitle(f"Model: {name} | FAR: {far:.4f}", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir,f"{name.replace(' ','_')}_eval.png"), dpi=120)
    plt.close()

    return {"model":name,"roc_auc":roc_auc,"false_alarm_rate":round(far,4),
            "tp":int(tp),"fp":int(fp),"fn":int(fn),"tn":int(tn),
            "precision":round(tp/(tp+fp+1e-8),4),"recall":round(tp/(tp+fn+1e-8),4)}


def train_all_detectors(X, y):
    X_normal = X[y==0]
    print("\n[→] Statistical Z-Score...")
    stat = StatisticalAnomalyDetector(); stat.fit(X_normal)
    print("[→] Isolation Forest...")
    iso  = IsolationForestDetector();   iso.fit(X_normal); iso.save("models/isolation_forest.pkl")
    print("[→] One-Class SVM...")
    idx  = np.random.choice(len(X_normal), min(800,len(X_normal)), replace=False)
    svm  = OneClassSVMDetector();       svm.fit(X_normal[idx]); svm.save("models/one_class_svm.pkl")

    results = [
        evaluate_detector("Statistical Z-Score", y, stat.predict(X), stat.score_samples(X)),
        evaluate_detector("Isolation Forest",     y, iso.predict(X),  iso.score_samples(X)),
        evaluate_detector("One-Class SVM",        y, svm.predict(X),  svm.score_samples(X)),
    ]

    # Save model comparison chart
    _plot_model_comparison(results)
    return stat, iso, svm, results


def _plot_model_comparison(results):
    import pandas as pd
    df = pd.DataFrame(results)
    fig, axes = plt.subplots(1,3,figsize=(14,5))
    metrics = ["roc_auc","precision","recall"]
    titles  = ["ROC-AUC","Precision","Recall"]
    colors  = ["#3b82f6","#10b981","#f59e0b"]
    for i,(m,title) in enumerate(zip(metrics,titles)):
        vals = df[m].fillna(0)
        bars = axes[i].bar(df["model"], vals, color=colors[i], alpha=0.85, edgecolor="white")
        axes[i].set_title(title, fontsize=13, fontweight="bold")
        axes[i].set_ylim(0,1.1)
        axes[i].set_xticklabels(df["model"], rotation=15, ha="right", fontsize=9)
        for bar,val in zip(bars,vals):
            axes[i].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                         f"{val:.3f}", ha="center", fontsize=10, fontweight="bold")
    plt.suptitle("Model Comparison Report", fontsize=14, fontweight="bold")
    plt.tight_layout()
    os.makedirs("reports", exist_ok=True)
    plt.savefig("reports/model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Model comparison chart → reports/model_comparison.png")
