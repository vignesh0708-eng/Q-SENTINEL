"""Train + benchmark the hybrid model on the Wisconsin Diagnostic Breast Cancer
dataset (bundled with scikit-learn, so no download is needed).

  python -m app.training

Models
  classical : logistic regression on all 30 features   (best classical baseline)
  classical4: logistic regression on the same 4 PCA features the circuit sees
  vqc       : 4-qubit data re-uploading variational circuit (trained here)
  hybrid    : advantage gate choosing between classical and vqc per patient
Positive class = malignant (disease). Metrics are recall-first.
"""
import json
import sys
import numpy as np
import joblib
from scipy.optimize import minimize
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, recall_score, precision_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from . import quantum as qc
from .config import MODEL_DIR, SCREEN_THRESHOLD
from .gate import advantage_gate

N_QUBITS, LAYERS = 4, 3
ARTIFACT = MODEL_DIR / "wdbc_hybrid.joblib"


def _vqc_loss(theta, X, y, sw):
    w = theta[:-2].reshape(LAYERS, N_QUBITS)
    p = qc.vqc_prob(X, w)
    prob = 1 / (1 + np.exp(-(theta[-2] * (p - theta[-1]))))
    prob = np.clip(prob, 1e-9, 1 - 1e-9)
    return float(-np.mean(sw * (y * np.log(prob) + (1 - y) * np.log(1 - prob))))


def _vqc_predict(theta, X):
    w = theta[:-2].reshape(LAYERS, N_QUBITS)
    p = qc.vqc_prob(X, w)
    return 1 / (1 + np.exp(-(theta[-2] * (p - theta[-1]))))


def train(seed: int = 42, restarts: int = 4, save: bool = True) -> dict:
    data = load_breast_cancer()
    X, y = data.data, (1 - data.target).astype(int)          # 1 = malignant
    names = list(data.feature_names)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)

    scaler = StandardScaler().fit(Xtr)
    pca = PCA(n_components=N_QUBITS, random_state=seed).fit(scaler.transform(Xtr))
    mm = MinMaxScaler().fit(pca.transform(scaler.transform(Xtr)))

    def encode(Xraw):
        z = mm.transform(pca.transform(scaler.transform(Xraw)))
        return np.clip(z, 0.0, 1.0)

    Ztr, Zte = encode(Xtr), encode(Xte)

    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(scaler.transform(Xtr), ytr)
    clf4 = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Ztr, ytr)

    sw = np.where(ytr == 1, 0.5 * len(ytr) / ytr.sum(), 0.5 * len(ytr) / (len(ytr) - ytr.sum()))
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(restarts):
        t0 = np.concatenate([rng.uniform(-np.pi, np.pi, LAYERS * N_QUBITS), [8.0, 0.5]])
        res = minimize(_vqc_loss, t0, args=(Ztr, ytr, sw), method="L-BFGS-B",
                       options={"maxiter": 120})
        if best is None or res.fun < best.fun:
            best = res
    theta = best.x

    c_te = clf.predict_proba(scaler.transform(Xte))[:, 1]
    c4_te = clf4.predict_proba(Zte)[:, 1]
    q_te = _vqc_predict(theta, Zte)
    h_te = np.array([advantage_gate(c, q)["risk"] for c, q in zip(c_te, q_te)])

    def metrics(p):
        yhat = (p >= SCREEN_THRESHOLD).astype(int)
        return {"auc": round(float(roc_auc_score(yte, p)), 4),
                "recall": round(float(recall_score(yte, yhat)), 4),
                "precision": round(float(precision_score(yte, yhat, zero_division=0)), 4),
                "accuracy": round(float(accuracy_score(yte, yhat)), 4)}

    bench = {
        "dataset": "Wisconsin Diagnostic Breast Cancer (UCI, via scikit-learn)",
        "n_train": int(len(ytr)), "n_test": int(len(yte)),
        "positive_class": "malignant", "screening_threshold": SCREEN_THRESHOLD,
        "qubits": N_QUBITS, "layers": LAYERS, "vqc_final_loss": round(float(best.fun), 4),
        "models": {"classical_30_features": metrics(c_te),
                   "classical_4_features": metrics(c4_te),
                   "quantum_vqc_4_qubits": metrics(q_te),
                   "hybrid_gate": metrics(h_te)},
        "honest_note": ("Simulator results on a small public dataset. This is a benchmark, "
                        "not evidence of clinical quantum advantage."),
    }
    artifact = {"scaler": scaler, "pca": pca, "mm": mm, "clf": clf, "theta": theta,
                "feature_names": names, "train_means": Xtr[ytr == 0].mean(axis=0),
                "benchmark": bench}
    if save:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, ARTIFACT)
    return artifact


class TrainedScreener:
    def __init__(self, artifact):
        self.a = artifact
        self.names = artifact["feature_names"]

    def _scores(self, Xraw):
        a = self.a
        z = np.clip(a["mm"].transform(a["pca"].transform(a["scaler"].transform(Xraw))), 0, 1)
        c = a["clf"].predict_proba(a["scaler"].transform(Xraw))[:, 1]
        q = _vqc_predict(a["theta"], z)
        return c, q

    def predict(self, features: dict, noise: float = 0.0) -> dict:
        x = np.array([[features[n] for n in self.names]], dtype=float)
        c, q = (v[0] for v in self._scores(x))
        g = advantage_gate(float(c), float(q), noise)
        # Attribution: set each feature to the benign-class mean, measure risk drop
        rows = []
        for i, n in enumerate(self.names):
            xb = x.copy()
            xb[0, i] = self.a["train_means"][i]
            cb, qb = (v[0] for v in self._scores(xb))
            rows.append({"feature": n, "delta": g["risk"] - advantage_gate(float(cb), float(qb), noise)["risk"]})
        rows.sort(key=lambda r: abs(r["delta"]), reverse=True)
        flag = g["risk"] >= SCREEN_THRESHOLD
        return {"risk": g["risk"], "risk_percent": round(g["risk"] * 100),
                "refer": bool(flag), "threshold": SCREEN_THRESHOLD,
                "classical_risk": float(c), "quantum_risk": float(q),
                "gate": {k: g[k] for k in ("mode", "classical_confidence", "quantum_confidence")},
                "top_contributors": [{**r, "delta_percent": round(r["delta"] * 100, 1)} for r in rows[:8]]}

    def sample(self, malignant: bool) -> dict:
        d = load_breast_cancer()
        idx = np.where(d.target == (0 if malignant else 1))[0][0]
        return {n: float(v) for n, v in zip(self.names, d.data[idx])}


def load_or_train() -> TrainedScreener:
    if ARTIFACT.exists():
        try:
            return TrainedScreener(joblib.load(ARTIFACT))
        except Exception:
            pass
    print("Training hybrid model (one-off, ~10-30 s)...", file=sys.stderr)
    return TrainedScreener(train())


if __name__ == "__main__":
    art = train()
    print(json.dumps(art["benchmark"], indent=2))
    print(f"\nSaved to {ARTIFACT}")
