import json, os, tempfile, unittest

os.environ["QSENTINEL_DB"] = os.path.join(tempfile.mkdtemp(), "t.db")

from app.main import create_app          # noqa: E402
from app import demo_model               # noqa: E402
from app.config import PRESETS           # noqa: E402


class Api(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = create_app(testing=True).test_client()

    def test_health(self):
        self.assertEqual(self.c.get("/api/health").get_json()["status"], "ok")

    def test_presets_order_and_bands(self):
        r = {k: demo_model.predict(v) for k, v in PRESETS.items()}
        self.assertLess(r["healthy"]["risk"], r["pre"]["risk"])
        self.assertLess(r["pre"]["risk"], r["high"]["risk"])
        self.assertEqual(r["healthy"]["band"], "Low")
        self.assertEqual(r["high"]["band"], "High")

    def test_matches_browser_reference(self):
        # Reference values produced by running the JavaScript in q-sentinel.html under Node
        ref = {("healthy", 0.0): (0.0727, 0.0373, "Quantum-led"),
               ("pre", 0.0): (0.4214, 0.2221, "Quantum-led"),
               ("high", 0.0): (0.9290, 0.9217, "Ensemble"),
               ("pre", 0.1): (0.4214, 0.2580, "Quantum-led")}
        for (name, noise), (c, q, mode) in ref.items():
            r = demo_model.predict(PRESETS[name], noise)
            self.assertAlmostEqual(r["classical_risk"], c, places=3)
            self.assertAlmostEqual(r["quantum_risk"], q, places=3)
            self.assertEqual(r["gate"]["mode"], mode)

    def test_predict_endpoint_and_save(self):
        body = dict(PRESETS["high"], noise=0.1, save=True, patient_ref="P1")
        j = self.c.post("/api/predict", json=body).get_json()
        self.assertIn("saved_id", j)
        self.assertEqual(len(j["contributions"]), 5)
        self.assertEqual(len(j["circuit"]["entanglers"]), 4)
        h = self.c.get("/api/history?patient_ref=P1").get_json()
        self.assertEqual(len(h), 1)
        self.assertEqual(self.c.delete(f"/api/history/{j['saved_id']}").status_code, 200)

    def test_noise_forces_classical_fallback(self):
        j = self.c.post("/api/predict", json=dict(PRESETS["pre"], noise=0.35)).get_json()
        self.assertEqual(j["gate"]["mode"], "Classical fallback")
        self.assertAlmostEqual(j["risk"], j["classical_risk"])

    def test_validation(self):
        r = self.c.post("/api/predict", json={"glucose": 9999})
        self.assertEqual(r.status_code, 422)
        self.assertEqual(self.c.post("/api/predict", data="x").status_code, 422)

    def test_trend_alert(self):
        for p in ("healthy", "high"):
            self.c.post("/api/predict", json=dict(PRESETS[p], save=True, patient_ref="T9"))
        t = self.c.get("/api/patients/T9/trend").get_json()
        self.assertTrue(t["alert"])

    def test_trained_model(self):
        s = self.c.get("/api/screen/wdbc/sample?malignant=true").get_json()["features"]
        j = self.c.post("/api/screen/wdbc", json={"features": s}).get_json()
        self.assertTrue(j["refer"])
        b = self.c.get("/api/benchmark").get_json()
        self.assertGreater(b["models"]["hybrid_gate"]["auc"], 0.95)
        self.assertEqual(self.c.post("/api/screen/wdbc", json={"features": {}}).status_code, 422)

    def test_index(self):
        self.assertEqual(self.c.get("/").status_code, 200)


if __name__ == "__main__":
    unittest.main()
