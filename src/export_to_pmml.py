from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from sklearn2pmml import sklearn2pmml
from sklearn2pmml.pipeline import PMMLPipeline

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
TARGET = "Survived"
FeatureKind = Literal["numeric", "categorical"]
IGNORED_COLUMNS = {"PassengerId", "Name", "Ticket", "Cabin"}


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    kind: FeatureKind


def infer_feature_specs(df: pd.DataFrame) -> list[FeatureSpec]:
    specs: list[FeatureSpec] = []
    for column in df.columns:
        if column == TARGET:
            continue
        series = df[column]
        if pd.api.types.is_numeric_dtype(series):
            specs.append(FeatureSpec(column, "numeric"))
            continue

        numeric_candidate = pd.to_numeric(series, errors="coerce")
        if numeric_candidate.notna().mean() >= 0.95:
            specs.append(FeatureSpec(column, "numeric"))
        else:
            specs.append(FeatureSpec(column, "categorical"))
    return specs


def split_feature_names(feature_specs: list[FeatureSpec], kind: FeatureKind) -> list[str]:
    return [spec.name for spec in feature_specs if spec.kind == kind]


def main() -> None:
    data_path = ROOT_DIR / "data" / "Titanic-Dataset.csv"
    model_path = ROOT_DIR / "models" / "titanic" / "random_forest_pipeline.pkl"
    output_path = ROOT_DIR / "model" / "titanic_random_forest.pmml"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Could not find trained model at {model_path}. Run src/train_random_forest.py first."
        )

    df = pd.read_csv(data_path)
    if TARGET not in df.columns:
        raise ValueError(f"{data_path} is missing target column '{TARGET}'.")

    df = df.dropna(subset=[TARGET])
    df = df.drop(columns=[col for col in IGNORED_COLUMNS if col in df.columns], errors="ignore")
    feature_specs = infer_feature_specs(df)
    feature_names = [spec.name for spec in feature_specs]

    X = df[feature_names].copy()
    y = df[TARGET]

    numeric_features = split_feature_names(feature_specs, "numeric")
    categorical_features = split_feature_names(feature_specs, "categorical")

    for col in numeric_features:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    for col in categorical_features:
        X[col] = X[col].astype("string").fillna("")

    X = X.dropna()  # 入力に欠損があると PMML 変換時に失敗するため除外
    y = y.loc[X.index]

    pipeline = joblib.load(model_path)

    pmml_pipeline = PMMLPipeline([
        ("pipeline", pipeline),
    ])
    pmml_pipeline.active_fields = feature_names
    pmml_pipeline.target_fields = [TARGET]

    # sklearn2pmml expects the pipeline to have fit attributes; ensure data is identical
    pmml_pipeline.fit(X, y)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sklearn2pmml(pmml_pipeline, output_path, with_repr=True)
    print(f"Exported PMML model to {output_path}")


if __name__ == "__main__":
    main()
