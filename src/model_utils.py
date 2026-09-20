from pathlib import Path
from pickle import load

import pandas as pd
from sklearn.pipeline import Pipeline


def make_inference(in_model: Pipeline, in_data: dict) -> dict[str, float]:
    prediction = in_model.predict(pd.DataFrame([in_data]))[0]
    return {"predicted_traffic_volume": round(float(prediction), 3)}


def load_model(path: str | Path) -> Pipeline:
    """Загрузка доверенного локального Pipeline из pickle."""
    with Path(path).open("rb") as file:
        return load(file)
