from pathlib import Path

import joblib
import pandas as pd
from sklearn2pmml import sklearn2pmml
from sklearn2pmml.pipeline import PMMLPipeline

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
NUMERIC_FEATURES = ["Age", "SibSp", "Parch", "Fare"]
TARGET = "Survived"


def main() -> None:
    data_path = ROOT_DIR / "data" / "Titanic-Dataset.csv"
    model_path = ROOT_DIR / "models" / "titanic" / "random_forest_pipeline.pkl"
    output_path = ROOT_DIR / "model" / "titanic_random_forest.pmml"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Could not find trained model at {model_path}. Run titanic/train_random_forest.py first."
        )

    df = pd.read_csv(data_path)
    X = df[FEATURES].copy()
    y = df[TARGET]

    for col in NUMERIC_FEATURES:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    for col in set(FEATURES) - set(NUMERIC_FEATURES):
        X[col] = X[col].astype(str)

    X = X.dropna()
    y = y.loc[X.index]

    pipeline = joblib.load(model_path)

    pmml_pipeline = PMMLPipeline([
        ("pipeline", pipeline),
    ])
    pmml_pipeline.active_fields = FEATURES
    pmml_pipeline.target_fields = [TARGET]

    # sklearn2pmml expects the pipeline to have fit attributes; ensure data is identical
    pmml_pipeline.fit(X, y)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sklearn2pmml(pmml_pipeline, output_path, with_repr=True)
    print(f"Exported PMML model to {output_path}")


if __name__ == "__main__":
    main()
