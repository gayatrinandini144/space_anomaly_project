"""
models/decision_engine.py
Phase 3 - Days 13-17: Autonomous Decision Layer

Covers:
- Rule-based vs AI-based decisions (Day 13)
- Simulated corrective actions (Day 15): shut subsystem, switch redundancy, alert
- Risk scoring system (Day 16)
- Decision confidence estimation (Day 17)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional
import json


# ── Enums & Data Classes ─────────────────────────────────────────────────────

class Subsystem(str, Enum):
    POWER        = "power"
    THERMAL      = "thermal"
    PROPULSION   = "propulsion"
    COMMUNICATION= "communication"
    OVERALL      = "overall"


class Action(str, Enum):
    NONE              = "no_action"
    ALERT             = "alert_ground_station"
    SHUTDOWN          = "shutdown_subsystem"
    SWITCH_REDUNDANCY = "switch_to_redundancy"
    REDUCE_POWER      = "reduce_power_mode"
    ENTER_SAFE_MODE   = "enter_safe_mode"
    REQUEST_HUMAN     = "request_human_review"


class Severity(str, Enum):
    NOMINAL  = "NOMINAL"
    CAUTION  = "CAUTION"
    WARNING  = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass
class Decision:
    timestamp: str
    subsystem: Subsystem
    anomaly_detected: bool
    severity: Severity
    risk_score: float
    confidence: float
    recommended_action: Action
    reasoning: str
    sensor_values: Dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "timestamp": self.timestamp,
            "subsystem": self.subsystem.value,
            "anomaly_detected": self.anomaly_detected,
            "severity": self.severity.value,
            "risk_score": round(self.risk_score, 4),
            "confidence": round(self.confidence, 4),
            "recommended_action": self.recommended_action.value,
            "reasoning": self.reasoning,
        }


# ── Rule-Based Decision System (Day 13) ─────────────────────────────────────

class RuleBasedDecisionSystem:
    """
    Deterministic threshold-based decision engine.
    Implements 'if sensor > threshold → action' logic.
    Simple, interpretable, zero latency.
    """

    THRESHOLDS = {
        "battery_charge":       {"caution": 40, "warning": 25, "critical": 15},
        "internal_temp":        {"caution": 35, "warning": 45, "critical": 60},
        "fuel_pressure":        {"caution": 130, "warning": 110, "critical": 90},  # low = bad
        "bit_error_rate":       {"caution": 0.01, "warning": 0.05, "critical": 0.1},
        "signal_strength":      {"caution": -70, "warning": -60, "critical": -50},  # high = bad
    }

    def evaluate(self, row: pd.Series, timestamp: str = "now") -> List[Decision]:
        decisions = []

        # Power subsystem
        batt = row.get("battery_charge", 100)
        if batt < self.THRESHOLDS["battery_charge"]["critical"]:
            decisions.append(Decision(
                timestamp=timestamp, subsystem=Subsystem.POWER,
                anomaly_detected=True, severity=Severity.CRITICAL,
                risk_score=0.95, confidence=0.98,
                recommended_action=Action.ENTER_SAFE_MODE,
                reasoning=f"Battery critically low: {batt:.1f}% (threshold 15%)",
                sensor_values={"battery_charge": batt}
            ))
        elif batt < self.THRESHOLDS["battery_charge"]["warning"]:
            decisions.append(Decision(
                timestamp=timestamp, subsystem=Subsystem.POWER,
                anomaly_detected=True, severity=Severity.WARNING,
                risk_score=0.70, confidence=0.95,
                recommended_action=Action.REDUCE_POWER,
                reasoning=f"Battery low: {batt:.1f}% – entering power-save mode",
                sensor_values={"battery_charge": batt}
            ))

        # Thermal subsystem
        temp = row.get("internal_temp", 22)
        if temp > self.THRESHOLDS["internal_temp"]["critical"]:
            decisions.append(Decision(
                timestamp=timestamp, subsystem=Subsystem.THERMAL,
                anomaly_detected=True, severity=Severity.CRITICAL,
                risk_score=0.92, confidence=0.97,
                recommended_action=Action.SHUTDOWN,
                reasoning=f"Thermal runaway detected: {temp:.1f}°C (limit 60°C)",
                sensor_values={"internal_temp": temp}
            ))

        # Propulsion subsystem
        fp = row.get("fuel_pressure", 150)
        if fp < self.THRESHOLDS["fuel_pressure"]["critical"]:
            decisions.append(Decision(
                timestamp=timestamp, subsystem=Subsystem.PROPULSION,
                anomaly_detected=True, severity=Severity.CRITICAL,
                risk_score=0.90, confidence=0.93,
                recommended_action=Action.SHUTDOWN,
                reasoning=f"Fuel pressure critically low: {fp:.1f} PSI – possible leak",
                sensor_values={"fuel_pressure": fp}
            ))

        # Communication subsystem
        ber = row.get("bit_error_rate", 0.001)
        if ber > self.THRESHOLDS["bit_error_rate"]["critical"]:
            decisions.append(Decision(
                timestamp=timestamp, subsystem=Subsystem.COMMUNICATION,
                anomaly_detected=True, severity=Severity.WARNING,
                risk_score=0.60, confidence=0.88,
                recommended_action=Action.SWITCH_REDUNDANCY,
                reasoning=f"Bit error rate {ber:.4f} exceeds limit – switching to backup antenna",
                sensor_values={"bit_error_rate": ber}
            ))

        if not decisions:
            decisions.append(Decision(
                timestamp=timestamp, subsystem=Subsystem.OVERALL,
                anomaly_detected=False, severity=Severity.NOMINAL,
                risk_score=0.02, confidence=0.99,
                recommended_action=Action.NONE,
                reasoning="All subsystems nominal. No action required.",
                sensor_values={}
            ))

        return decisions


# ── AI-Based Decision System (Day 13) ────────────────────────────────────────

class AIDecisionSystem:
    """
    ML-augmented decision engine that combines anomaly scores
    with risk scoring and confidence estimation.
    Day 13: AI-based decisions vs rule-based.
    Day 16: Risk scoring.
    Day 17: Decision confidence estimation.
    """

    def __init__(self, anomaly_models: Dict = None):
        self.anomaly_models = anomaly_models or {}

    def _compute_ensemble_risk(self, scores: Dict[str, float]) -> tuple:
        """
        Average anomaly scores from multiple models → ensemble risk.
        Confidence = 1 - variance across models.
        """
        if not scores:
            return 0.1, 0.5

        vals = np.array(list(scores.values()))
        # Normalize scores to [0, 1] — higher = more anomalous
        vals = (vals - vals.min()) / (vals.ptp() + 1e-8)
        risk = float(vals.mean())
        variance = float(vals.std())
        confidence = float(np.clip(1 - variance * 2, 0.3, 0.99))
        return risk, confidence

    def _risk_to_severity(self, risk: float) -> Severity:
        if risk < 0.25:   return Severity.NOMINAL
        if risk < 0.50:   return Severity.CAUTION
        if risk < 0.75:   return Severity.WARNING
        return Severity.CRITICAL

    def _risk_to_action(self, risk: float, subsystem: Subsystem,
                        confidence: float) -> Action:
        if risk < 0.25:
            return Action.NONE
        if risk < 0.50:
            return Action.ALERT
        if confidence < 0.6:
            return Action.REQUEST_HUMAN    # low confidence → human review
        if risk < 0.75:
            return Action.SWITCH_REDUNDANCY
        if risk < 0.90:
            return Action.REDUCE_POWER
        return Action.ENTER_SAFE_MODE

    def evaluate(self, row: pd.Series, timestamp: str = "now",
                 anomaly_scores: Dict = None) -> Decision:
        """
        Generate an AI-based decision for a single telemetry reading.
        """
        anomaly_scores = anomaly_scores or {}
        risk, confidence = self._compute_ensemble_risk(anomaly_scores)

        # Adjust risk using mission health scores if available
        mission_risk = row.get("mission_risk_score", risk)
        final_risk = 0.6 * risk + 0.4 * float(mission_risk)

        severity = self._risk_to_severity(final_risk)
        subsystem = Subsystem.OVERALL
        action = self._risk_to_action(final_risk, subsystem, confidence)

        reasoning = (
            f"Ensemble risk score: {final_risk:.3f} | "
            f"Models: {list(anomaly_scores.keys())} | "
            f"Confidence: {confidence:.2%} | "
            f"Action: {action.value}"
        )

        return Decision(
            timestamp=timestamp, subsystem=subsystem,
            anomaly_detected=(final_risk > 0.25),
            severity=severity, risk_score=final_risk,
            confidence=confidence, recommended_action=action,
            reasoning=reasoning,
        )


# ── Decision Logger ───────────────────────────────────────────────────────────

class DecisionLogger:
    """Logs decisions to JSON for dashboard display."""

    def __init__(self, log_path="reports/decision_log.json"):
        self.log_path = log_path
        self.log = []

    def record(self, decision: Decision):
        self.log.append(decision.to_dict())

    def save(self):
        import os
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        with open(self.log_path, "w") as f:
            json.dump(self.log, f, indent=2, default=str)
        print(f"✅ Decision log saved → {self.log_path} ({len(self.log)} entries)")

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame(self.log)


# ── Simulation ────────────────────────────────────────────────────────────────

def simulate_mission(df: pd.DataFrame, n_steps: int = 100) -> pd.DataFrame:
    """
    Run autonomous decision simulation on telemetry stream.
    Combines rule-based and AI-based approaches.
    """
    rule_engine = RuleBasedDecisionSystem()
    ai_engine   = AIDecisionSystem()
    logger      = DecisionLogger()

    # Use a random subsample for the simulation
    sample = df.sample(min(n_steps, len(df))).reset_index(drop=True)

    for i, row in sample.iterrows():
        ts = str(row.get("timestamp", f"T+{i}"))

        # Rule-based decisions
        rule_decisions = rule_engine.evaluate(row, ts)
        for d in rule_decisions:
            logger.record(d)

        # AI-based decision (uses health scores already in the row)
        mock_scores = {
            "isolation_forest": float(np.random.beta(2, 5)),
            "autoencoder":      float(np.random.beta(2, 5)),
        }
        if row.get("anomaly_label", 0) == 1:
            mock_scores = {k: v * 3 for k, v in mock_scores.items()}  # amplify

        ai_decision = ai_engine.evaluate(row, ts, anomaly_scores=mock_scores)
        logger.record(ai_decision)

    logger.save()
    return logger.summary()


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    from utils.data_generator import generate_spacecraft_telemetry, add_rul_labels
    from utils.feature_engineering import (
        handle_missing_noisy_data, add_subsystem_health_scores
    )

    df = generate_spacecraft_telemetry(500)
    df = add_rul_labels(df)
    df = handle_missing_noisy_data(df)
    df = add_subsystem_health_scores(df)

    summary = simulate_mission(df, n_steps=100)
    print(summary["severity"].value_counts())
    print(summary["recommended_action"].value_counts())
