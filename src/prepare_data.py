"""Charge data/raw/admission.csv, le nettoie et le découpe en jeux d'entraînement et de test.

Sorties : data/processed/X_train.csv, X_test.csv, y_train.csv, y_test.csv
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_FILE = BASE_DIR / "data" / "raw" / "admission.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

TARGET = "Chance of Admit"
TEST_SIZE = 0.2
RANDOM_STATE = 42


def main() -> None:
    df = pd.read_csv(RAW_FILE)

    # Les noms de colonnes du fichier source contiennent des espaces en trop ("LOR ", "Chance of Admit ")
    df.columns = df.columns.str.strip()
    # Le numéro de ligne n'apporte aucune information pour la prédiction
    df = df.drop(columns=["Serial No."], errors="ignore")
    df = df.dropna().drop_duplicates()

    X = df.drop(columns=TARGET)
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(PROCESSED_DIR / "X_train.csv", index=False)
    X_test.to_csv(PROCESSED_DIR / "X_test.csv", index=False)
    y_train.to_csv(PROCESSED_DIR / "y_train.csv", index=False)
    y_test.to_csv(PROCESSED_DIR / "y_test.csv", index=False)

    print(f"Données : {len(df)} lignes, variables : {', '.join(X.columns)}")
    print(f"Entraînement : {len(X_train)} lignes, test : {len(X_test)} lignes")


if __name__ == "__main__":
    main()
