# 🛰️ AI Space Anomaly Detection — Enhanced Edition

## 📁 Project Structure
```
space_enhanced/
├── train.py
├── requirements.txt
├── utils/
│   ├── data_generator.py        ← Telemetry simulation + NASA loader
│   └── feature_engineering.py
├── models/
│   ├── anomaly_detection.py     ← Z-Score, Isolation Forest, One-Class SVM
│   ├── autoencoder.py           ← SVD Autoencoder
│   ├── lstm_model.py            ← LSTM RUL predictor
│   └── decision_engine.py       ← Rule-based + AI decisions (ptp bug fixed)
└── dashboard/
    └── app.py                   ← 7-page enhanced dashboard
```

## 🆕 What's New (Enhanced Edition)
- ✅ ptp() bug fixed (works with latest NumPy)
- 🏠 Home Dashboard with KPIs + live alerts
- 📈 Live Telemetry with anomaly type breakdown
- 🔮 Anomaly Predictor — enter sensor values, get instant prediction
- 📊 Model Comparison Report with radar chart
- 🤖 Decision Log with filters
- 🎮 Mission Simulator game — trigger anomalies, apply fixes!
- 🛸 NASA Dataset page with automatic detection

## ⚙️ Setup
```bash
py -3.11 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## ▶️ Run
```bash
python train.py --phase 1
python train.py --phase 2
python train.py --phase 3
streamlit run dashboard/app.py
```
Open: http://localhost:8501

## 🛸 NASA Dataset (Optional)
1. Download from https://www.kaggle.com/datasets/behrad3d/nasa-cmaps
2. Create folder: data/nasa_cmapss/
3. Place train_FD001.txt inside
4. Re-run: python train.py --phase 1
