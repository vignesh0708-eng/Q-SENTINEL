"""Flask application factory + routes."""
import time
from flask import Flask, jsonify, request, send_from_directory
from pydantic import BaseModel, Field, ValidationError, create_model
from typing import Optional

from . import db, demo_model
from .config import API_KEY, FEATURES, PRESETS, STATIC_DIR, NOISE_FALLBACK
from .training import load_or_train, ARTIFACT

_fields = {f["id"]: (float, Field(ge=f["min"], le=f["max"])) for f in FEATURES}


class _Base(BaseModel):
    noise: float = Field(0.0, ge=0.0, le=0.4, description="0-0.4 (0-40 %)")
    save: bool = False
    patient_ref: Optional[str] = Field(None, max_length=64)


PredictRequest = create_model("PredictRequest", __base__=_Base, **_fields)


def create_app(testing: bool = False) -> Flask:
    app = Flask(__name__, static_folder=None)
    app.json.sort_keys = False
    app.config["TESTING"] = testing
    screener = {"m": None}

    def get_screener():
        if screener["m"] is None:
            screener["m"] = load_or_train()
        return screener["m"]

    # ------------------------------------------------------------ plumbing
    @app.after_request
    def cors(resp):
        resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type, X-API-Key"
        resp.headers["Access-Control-Allow-Methods"] = "GET, POST, DELETE, OPTIONS"
        return resp

    @app.before_request
    def guard():
        if request.method == "OPTIONS":
            return ("", 204)
        if API_KEY and request.method in ("POST", "DELETE") and request.path.startswith("/api/"):
            if request.headers.get("X-API-Key") != API_KEY:
                return jsonify(error="invalid or missing X-API-Key"), 401

    @app.errorhandler(ValidationError)
    def bad_input(e):
        return jsonify(error="validation failed",
                       details=[{"field": ".".join(map(str, x["loc"])), "message": x["msg"]}
                                for x in e.errors()]), 422

    @app.errorhandler(404)
    def nf(_):
        return jsonify(error="not found"), 404

    @app.errorhandler(500)
    def boom(_):
        return jsonify(error="internal error"), 500

    def body():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            raise ValidationError.from_exception_data("body", [
                {"type": "dict_type", "loc": ("body",), "input": data}])
        return data

    # --------------------------------------------------------------- pages
    @app.get("/")
    def index():
        return send_from_directory(STATIC_DIR, "index.html")

    # ----------------------------------------------------------------- API
    @app.get("/api/health")
    def health():
        return jsonify(status="ok", time=time.time(), trained_model=ARTIFACT.exists(),
                       api_key_required=bool(API_KEY))

    @app.get("/api/config")
    def config():
        return jsonify(features=FEATURES, presets=PRESETS, noise_fallback_threshold=NOISE_FALLBACK)

    @app.post("/api/predict")
    def predict():
        req = PredictRequest(**body())
        values = {f["id"]: getattr(req, f["id"]) for f in FEATURES}
        result = demo_model.predict(values, req.noise)
        if req.save:
            result["saved_id"] = db.save("demo", req.patient_ref, {**values, "noise": req.noise}, result)
        return jsonify(result)

    @app.get("/api/predict/preset/<name>")
    def preset(name):
        if name not in PRESETS:
            return jsonify(error=f"unknown preset; choose from {list(PRESETS)}"), 404
        return jsonify(inputs=PRESETS[name], result=demo_model.predict(PRESETS[name]))

    @app.post("/api/screen/wdbc")
    def screen_wdbc():
        m = get_screener()
        data = body()
        noise = float(data.get("noise", 0.0))
        feats = data.get("features")
        if not isinstance(feats, dict):
            return jsonify(error="send {'features': {<30 WDBC feature names>: number}}",
                           required=m.names), 422
        missing = [n for n in m.names if n not in feats]
        if missing:
            return jsonify(error="missing features", missing=missing), 422
        try:
            feats = {n: float(feats[n]) for n in m.names}
        except (TypeError, ValueError):
            return jsonify(error="all feature values must be numbers"), 422
        result = m.predict(feats, min(max(noise, 0.0), 0.4))
        if data.get("save"):
            result["saved_id"] = db.save("wdbc", data.get("patient_ref"), feats, {**result, "band": "Refer" if result["refer"] else "Routine"})
        return jsonify(result)

    @app.get("/api/screen/wdbc/sample")
    def wdbc_sample():
        m = get_screener()
        malignant = request.args.get("malignant", "true").lower() != "false"
        return jsonify(features=m.sample(malignant))

    @app.get("/api/benchmark")
    def benchmark():
        return jsonify(get_screener().a["benchmark"])

    @app.post("/api/benchmark/retrain")
    def retrain():
        from .training import train
        screener["m"] = type(get_screener())(train(seed=int(request.args.get("seed", 42))))
        return jsonify(screener["m"].a["benchmark"])

    @app.get("/api/history")
    def history():
        limit = max(1, min(int(request.args.get("limit", 50)), 500))
        return jsonify(db.history(limit, request.args.get("patient_ref")))

    @app.get("/api/history/<int:i>")
    def history_one(i):
        r = db.get(i)
        return (jsonify(r), 200) if r else (jsonify(error="not found"), 404)

    @app.delete("/api/history/<int:i>")
    def history_del(i):
        return (jsonify(deleted=i), 200) if db.delete(i) else (jsonify(error="not found"), 404)

    @app.get("/api/patients/<ref>/trend")
    def trend(ref):
        return jsonify(db.trend(ref))

    return app
