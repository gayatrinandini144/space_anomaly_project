"""
models/decision_engine.py — Autonomous Decision Layer (Day 13-17)
"""
import numpy as np
import pandas as pd
import json, os

SEVERITY = {"NOMINAL":0,"CAUTION":1,"WARNING":2,"CRITICAL":3}

def _make(ts, sub, sev, risk, conf, action, reason):
    return {"timestamp":str(ts),"subsystem":sub,"severity":sev,
            "risk_score":round(risk,4),"confidence":round(conf,4),
            "recommended_action":action,"reasoning":reason,
            "anomaly_detected": sev!="NOMINAL"}


class RuleBasedDecisionSystem:
    LIMITS = {
        "battery_charge": {"caution":40,"warning":25,"critical":15},
        "internal_temp":  {"caution":35,"warning":45,"critical":60},
        "fuel_pressure":  {"caution":130,"warning":110,"critical":90},
        "bit_error_rate": {"caution":0.01,"warning":0.05,"critical":0.1},
    }
    def evaluate(self, row, ts="T"):
        out = []
        b = float(row.get("battery_charge",100))
        if   b < self.LIMITS["battery_charge"]["critical"]:
            out.append(_make(ts,"POWER","CRITICAL",0.95,0.98,"enter_safe_mode",
                             f"Battery critical: {b:.1f}%"))
        elif b < self.LIMITS["battery_charge"]["warning"]:
            out.append(_make(ts,"POWER","WARNING",0.70,0.95,"reduce_power_mode",
                             f"Battery low: {b:.1f}%"))

        tmp = float(row.get("internal_temp",22))
        if   tmp > self.LIMITS["internal_temp"]["critical"]:
            out.append(_make(ts,"THERMAL","CRITICAL",0.92,0.97,"shutdown_subsystem",
                             f"Thermal runaway: {tmp:.1f}°C"))
        elif tmp > self.LIMITS["internal_temp"]["warning"]:
            out.append(_make(ts,"THERMAL","WARNING",0.65,0.90,"alert_ground_station",
                             f"High temp: {tmp:.1f}°C"))

        fp = float(row.get("fuel_pressure",150))
        if fp < self.LIMITS["fuel_pressure"]["critical"]:
            out.append(_make(ts,"PROPULSION","CRITICAL",0.90,0.93,"shutdown_subsystem",
                             f"Fuel pressure critical: {fp:.1f} PSI"))

        ber = float(row.get("bit_error_rate",0.001))
        if ber > self.LIMITS["bit_error_rate"]["critical"]:
            out.append(_make(ts,"COMM","WARNING",0.60,0.88,"switch_to_redundancy",
                             f"High bit error rate: {ber:.5f}"))

        if not out:
            out.append(_make(ts,"OVERALL","NOMINAL",0.02,0.99,"no_action",
                             "All systems nominal"))
        return out


class AIDecisionSystem:
    def evaluate(self, row, ts="T", anomaly_scores=None):
        scores = anomaly_scores or {}
        if scores:
            vals      = np.array(list(scores.values()), dtype=float)
            vmin, vmax = vals.min(), vals.max()
            vals_norm  = (vals - vmin) / ((vmax - vmin) + 1e-8)   # fixed: no ptp()
            risk       = float(vals_norm.mean())
            confidence = float(np.clip(1 - vals_norm.std()*2, 0.3, 0.99))
        else:
            risk, confidence = 0.1, 0.5

        mission_risk = float(row.get("mission_risk_score", risk))
        final_risk   = round(0.6*risk + 0.4*mission_risk, 4)

        sev    = ("CRITICAL" if final_risk>0.75 else "WARNING" if final_risk>0.50
                  else "CAUTION" if final_risk>0.25 else "NOMINAL")
        action = ("enter_safe_mode"      if final_risk>0.75 else
                  "request_human_review" if final_risk>0.50 and confidence<0.6 else
                  "switch_to_redundancy" if final_risk>0.50 else
                  "alert_ground_station" if final_risk>0.25 else "no_action")

        return _make(ts,"OVERALL",sev,final_risk,confidence,action,
                     f"Ensemble risk={final_risk:.3f} | conf={confidence:.0%} | "
                     f"models={list(scores.keys())}")


def simulate_mission(df, n_steps=300):
    rule = RuleBasedDecisionSystem()
    ai   = AIDecisionSystem()
    log  = []
    sample = df.sample(min(n_steps,len(df)), random_state=42).reset_index(drop=True)
    for _, row in sample.iterrows():
        ts = str(row.get("timestamp","T"))
        for d in rule.evaluate(row, ts):
            log.append(d)
        base = np.random.beta(2,5)
        if row.get("anomaly_label",0)==1:
            base = min(base*3.5, 1.0)
        ai_scores = {"isolation_forest": base,
                     "autoencoder": min(base*np.random.uniform(0.8,1.2),1.0)}
        log.append(ai.evaluate(row, ts, anomaly_scores=ai_scores))

    os.makedirs("reports", exist_ok=True)
    with open("reports/decision_log.json","w") as f:
        json.dump(log, f, indent=2, default=str)
    print(f"  Decision log → reports/decision_log.json ({len(log)} entries)")
    return pd.DataFrame(log)
