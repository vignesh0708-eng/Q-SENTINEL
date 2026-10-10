"""Advantage gate: decide whether the quantum or classical score is used."""
from .config import NOISE_FALLBACK, GATE_MARGIN


def advantage_gate(c: float, q: float, noise: float = 0.0) -> dict:
    cc, qc = abs(c - 0.5) * 2, abs(q - 0.5) * 2
    if noise > NOISE_FALLBACK:
        mode, risk = "Classical fallback", c
    elif qc > cc + GATE_MARGIN:
        mode, risk = "Quantum-led", q
    elif cc > qc + GATE_MARGIN:
        mode, risk = "Classical-led", c
    else:
        mode, risk = "Ensemble", (c + q) / 2
    return {"mode": mode, "risk": risk, "classical_confidence": cc, "quantum_confidence": qc}


def band_of(r: float) -> dict:
    if r < 0.3:
        return {"label": "Low", "action": "Routine annual check-up."}
    if r < 0.6:
        return {"label": "Moderate", "action": "Schedule a screening test within 3 months."}
    return {"label": "High", "action": "Refer for clinical evaluation promptly."}
