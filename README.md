# 🛰️ AI-Based Real-Time Anomaly Detection & Autonomous Decision Support System
### For Deep Space Missions — 21-Day Project

---

## 📁 Project Structure

```
space_anomaly_project/
│
├── train.py                        ← Master pipeline (run this first)
├── requirements.txt                ← All Python dependencies
│
├── utils/
│   ├── data_generator.py           ← Day 1-2: Simulates spacecraft telemetry data
│   └── feature_engineering.py      ← Day 2, 6-7: Feature extraction & health scores
│
├── models/
│   ├── anomaly_detection.py        ← Day 3, 8-9: Isolation Forest, One-Class SVM, Z-Score
│   ├── autoencoder.py              ← Day 10: Deep Learning Autoencoder
│   ├── lstm_model.py               ← Day 4, 11: LSTM + Remaining Useful Life (RUL)
│   └── decision_engine.py          ← Day 13-17: Rule-based + AI autonomous decisions
│
├── dashboard/
│   └── app.py                      ← Day 18-20: Streamlit Mission Control Dashboard
│
├── data/                           ← Auto-created after running train.py
│   └── spacecraft_telemetry.csv
│
├── models/ (saved)                 ← Auto-created after training
│   ├── isolation_forest.pkl
│   ├── one_class_svm.pkl
│   ├── scaler.pkl
│   ├── autoencoder/
│   └── lstm/
│
└── reports/                        ← Auto-created after training
    ├── decision_log.json
    ├── decision_summary.csv
    └── *.png  (evaluation plots)
```

---

## ⚙️ Setup Instructions

### Step 1 — Install Python (3.9 or higher)
Download from https://www.python.org/downloads/

### Step 2 — Create a virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

> ⚠️ TensorFlow is optional. If it fails to install, the project automatically
> uses NumPy fallback implementations for the Autoencoder and LSTM.

---

## 🚀 How to Run

### Option A — Run everything at once
```bash
python train.py
```

### Option B — Run phase by phase (recommended for learning)
```bash
python train.py --phase 1    # Generate telemetry data (Day 1-5)
python train.py --phase 2    # Train all ML models (Day 6-12)
python train.py --phase 3    # Run autonomous decision engine (Day 13-17)
```

### Option C — Run the Dashboard (Day 18-20)
```bash
streamlit run dashboard/app.py
```
Then open http://localhost:8501 in your browser.

---

---

## 📊 Sensors Simulated

| Subsystem | Sensors |
|-----------|---------|
| ⚡ Power | Solar panel voltage, Battery charge, Power consumption |
| 🌡️ Thermal | Internal temp, External temp, Thermal control power |
| 🔥 Propulsion | Fuel pressure, Thruster temp, Delta-V |
| 📡 Communication | Signal strength, Data rate, Bit error rate |

---
---

## 💡 Tips

- Start with `train.py --phase 1` to understand the data before touching models
- Open each `.py` file and read the comments — every function maps to a day
- The dashboard works even before training (generates live demo data automatically)
- For the report, compare all 3 models using the plots in `reports/`
