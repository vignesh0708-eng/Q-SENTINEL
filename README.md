# Q-Sentinel backend

REST API + web dashboard for the Q-Sentinel hybrid quantum-classical early disease screening demo.
Pure Python (Flask + NumPy + scikit-learn). The 5-qubit circuit is simulated exactly with NumPy,
so **no quantum SDK or hardware is needed**.

## Run

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run.py
```
Open http://127.0.0.1:8000 (dashboard) or http://127.0.0.1:8000/api/health.
On first use of the trained-model endpoints a model is trained (~10 s) and cached in `models/`.
Shortcuts: `./run.sh` (Linux/Mac) or `run.bat` (Windows). Docker: `docker build -t qsentinel . && docker run -p 8000:8000 qsentinel`.

## What is inside

| Part | Where | What it does |
|---|---|---|
| Demo model | `app/demo_model.py` | Classical logistic + 5-qubit circuit + advantage gate + ablation attribution. Matches `q-sentinel.html` to ~1e-16 |
| Simulator | `app/quantum.py` | NumPy state-vector (RY, CRY) and a batched VQC simulator (RY, CNOT) |
| Advantage gate | `app/gate.py` | Quantum-led / Classical-led / Ensemble / Classical fallback when noise > 25% |
| Trained model | `app/training.py` | 4-qubit data re-uploading VQC trained on Wisconsin breast cancer data (built into scikit-learn), benchmarked against classical baselines, recall-first threshold 0.40 |
| History | `app/db.py` | SQLite log of screenings and per-patient risk trend / drift alert |
| API | `app/main.py` | Routes below |

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | liveness |
| GET | `/api/config` | feature ranges, presets |
| POST | `/api/predict` | `{glucose,bmi,bp,chol,age,noise?,save?,patient_ref?}` -> risk, band, classical vs quantum, gate, attribution, circuit |
| GET | `/api/predict/preset/{healthy\|pre\|high}` | ready-made example |
| POST | `/api/screen/wdbc` | `{features:{30 WDBC names:number}}` -> trained hybrid risk + top contributors |
| GET | `/api/screen/wdbc/sample?malignant=true` | sample input for the call above |
| GET | `/api/benchmark` | AUC / recall / precision for classical, VQC and hybrid gate |
| POST | `/api/benchmark/retrain?seed=42` | retrain |
| GET | `/api/history?limit=&patient_ref=` | saved screenings |
| GET / DELETE | `/api/history/{id}` | one record |
| GET | `/api/patients/{ref}/trend` | risk over time, `alert` if risk rose > 10 points |

```bash
curl -X POST localhost:8000/api/predict -H "Content-Type: application/json" \
  -d '{"glucose":190,"bmi":37,"bp":168,"chol":280,"age":68,"noise":0.1,"save":true,"patient_ref":"P-001"}'
```

Errors are JSON (`422` for bad input with per-field messages). Set `QSENTINEL_API_KEY` to require an
`X-API-Key` header on POST/DELETE. Other env vars: `HOST`, `PORT`, `QSENTINEL_DB`.

## Tests
```bash
python -m unittest discover -s tests -v
```

## Honest limitations
* The 5-indicator demo uses **illustrative, hand-set weights** (as in the HTML), not trained on patient data.
* The trained model is on a small public dataset, simulator only. In the benchmark the quantum circuit
  does **not** beat the classical baseline; the hybrid gate sits in between. That matches the caution on
  slide 4 of the deck: report it as a benchmark, not as clinical quantum advantage.
* Not a medical device. For production, use gunicorn/waitress behind HTTPS, add real auth, and restrict CORS.
