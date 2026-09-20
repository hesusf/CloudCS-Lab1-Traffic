import json
import math
from pathlib import Path

from model_utils import load_model, make_inference

ROOT = Path(__file__).resolve().parents[1]


def test_unknown_category():
    model = load_model(ROOT / "models/pipeline.pkl")
    data = json.loads((ROOT / "data/example_request.json").read_text())
    data["weather_main"] = "Unknown category"
    result = make_inference(model, data)
    value = result["predicted_traffic_volume"]
    assert isinstance(value, (int, float))
    assert math.isfinite(value) and value >= 0
