"""Entraîne une régression linéaire (avec normalisation), l'évalue sur le jeu de test
et l'enregistre dans le Model Store de BentoML sous le nom admission_lr.
"""

from pathlib import Path

import bentoml
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODEL_NAME = "admission_lr"


def load_split(name: str) -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / f"{name}.csv")


def main() -> None:
    X_train, X_test = load_split("X_train"), load_split("X_test")
    y_train = load_split("y_train").squeeze("columns")
    y_test = load_split("y_test").squeeze("columns")

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", LinearRegression()),
    ])
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "r2": float(r2_score(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "mae": float(mean_absolute_error(y_test, y_pred)),
    }
    print("Performances sur le jeu de test : "
          + ", ".join(f"{name.upper()} = {value:.4f}" for name, value in metrics.items()))

    saved_model = bentoml.sklearn.save_model(
        MODEL_NAME,
        model,
        signatures={"predict": {"batchable": False}},
        metadata={**metrics, "features": list(X_train.columns)},
    )
    print(f"Modèle enregistré dans le Model Store de BentoML : {saved_model.tag}")


if __name__ == "__main__":
    main()
