"""The 5-indicator demo model (classical logistic + 5-qubit circuit + gate).
Line-for-line equivalent of the JavaScript in q-sentinel.html."""
import math
from . import quantum as qc
from .config import (FEATURES, CLASSIC_BIAS, ENC_SCALE, COUPLE, Q_SHARPNESS,
                     Q_BIAS, BASELINE_X)
from .gate import advantage_gate, band_of

N = len(FEATURES)


def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def normalise(values: dict) -> list:
    out = []
    for f in FEATURES:
        v = values[f["id"]]
        out.append(min(1.0, max(0.0, (v - f["min"]) / (f["max"] - f["min"]))))
    return out


def classical(xs) -> float:
    z = CLASSIC_BIAS + sum(f["wc"] * xs[i] for i, f in enumerate(FEATURES))
    return sigmoid(z)


def quantum(xs, noise: float = 0.0) -> dict:
    s = qc.new_state(N)
    for i, f in enumerate(FEATURES):
        qc.apply_ry(s, f["qubit"], ENC_SCALE * xs[i])
    for q in range(N - 1):
        qc.apply_cry(s, q, q + 1, COUPLE)
    p = [qc.prob_one(s, f["qubit"]) * (1 - noise) + 0.5 * noise for f in FEATURES]
    score = sum(f["wq"] * p[i] for i, f in enumerate(FEATURES))
    return {"risk": sigmoid(Q_SHARPNESS * (score - Q_BIAS)), "p": p}


def hybrid(xs, noise: float = 0.0) -> dict:
    c = classical(xs)
    q = quantum(xs, noise)["risk"]
    g = advantage_gate(c, q, noise)
    g.update({"classical": c, "quantum": q})
    return g


def attribution(xs, noise: float, final_risk: float) -> list:
    """Ablation: drop in final risk if one indicator were set to a healthy baseline."""
    rows = []
    for i, f in enumerate(FEATURES):
        xb = list(xs)
        xb[i] = min(xs[i], BASELINE_X)
        rows.append({"id": f["id"], "name": f["name"],
                     "delta": final_risk - hybrid(xb, noise)["risk"]})
    rows.sort(key=lambda r: abs(r["delta"]), reverse=True)
    return rows


def circuit_description(xs, p) -> dict:
    return {
        "qubits": N,
        "encoding": [{"qubit": f["qubit"], "feature": f["id"], "gate": "RY",
                      "angle_pi": round(ENC_SCALE * xs[i] / math.pi, 4),
                      "p_one": round(p[i], 4)} for i, f in enumerate(FEATURES)],
        "entanglers": [{"control": q, "target": q + 1, "gate": "CRY",
                        "angle_pi": round(COUPLE / math.pi, 4)} for q in range(N - 1)],
    }


def predict(values: dict, noise: float = 0.0) -> dict:
    xs = normalise(values)
    h = hybrid(xs, noise)
    qr = quantum(xs, noise)
    band = band_of(h["risk"])
    return {
        "risk": h["risk"],
        "risk_percent": round(h["risk"] * 100),
        "band": band["label"],
        "action": band["action"],
        "classical_risk": h["classical"],
        "quantum_risk": h["quantum"],
        "gate": {"mode": h["mode"],
                 "classical_confidence": h["classical_confidence"],
                 "quantum_confidence": h["quantum_confidence"],
                 "noise": noise},
        "contributions": [{**r, "delta_percent": round(r["delta"] * 100, 1)}
                          for r in attribution(xs, noise, h["risk"])],
        "qubit_probabilities": {f["id"]: qr["p"][i] for i, f in enumerate(FEATURES)},
        "circuit": circuit_description(xs, qr["p"]),
        "disclaimer": "Demo only. Illustrative weights, not a medical device.",
    }
