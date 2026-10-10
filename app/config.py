"""Configuration shared by the whole backend.

FEATURES / constants mirror q-sentinel.html exactly so the backend and the
browser demo give the same numbers for the same inputs.
"""
import math
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
DB_PATH = Path(os.environ.get("QSENTINEL_DB", BASE_DIR / "qsentinel.db"))
STATIC_DIR = BASE_DIR / "static"

# Optional API key. If QSENTINEL_API_KEY is set, POST/DELETE need header X-API-Key.
API_KEY = os.environ.get("QSENTINEL_API_KEY", "")

FEATURES = [
    {"id": "glucose", "name": "Fasting glucose",   "unit": "mg/dL", "min": 70,  "max": 250, "default": 95,  "qubit": 0, "wc": 3.0, "wq": 0.30},
    {"id": "bmi",     "name": "BMI",               "unit": "kg/m2", "min": 15,  "max": 45,  "default": 23,  "qubit": 1, "wc": 1.4, "wq": 0.15},
    {"id": "bp",      "name": "Systolic BP",       "unit": "mmHg",  "min": 90,  "max": 200, "default": 118, "qubit": 2, "wc": 1.8, "wq": 0.20},
    {"id": "chol",    "name": "Total cholesterol", "unit": "mg/dL", "min": 120, "max": 320, "default": 170, "qubit": 3, "wc": 1.2, "wq": 0.15},
    {"id": "age",     "name": "Age",               "unit": "years", "min": 20,  "max": 90,  "default": 35,  "qubit": 4, "wc": 2.2, "wq": 0.20},
]

PRESETS = {
    "healthy": {"glucose": 90,  "bmi": 22, "bp": 115, "chol": 165, "age": 30},
    "pre":     {"glucose": 128, "bmi": 29, "bp": 132, "chol": 215, "age": 52},
    "high":    {"glucose": 190, "bmi": 37, "bp": 168, "chol": 280, "age": 68},
}

CLASSIC_BIAS = -4.2
ENC_SCALE = 0.8 * math.pi      # angle-encoding range
COUPLE = 0.4                   # controlled-RY coupling angle
Q_SHARPNESS = 9.0
Q_BIAS = 0.42
BASELINE_X = 0.2               # "healthy" normalised level for attribution
NOISE_FALLBACK = 0.25          # above this noise the gate always uses classical
GATE_MARGIN = 0.05

# Screening threshold for the trained model (recall-first: below 0.5 on purpose)
SCREEN_THRESHOLD = 0.40
