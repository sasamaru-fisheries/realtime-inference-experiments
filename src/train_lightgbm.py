import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import joblib
import lightgbm as lgb
import optuna
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
# スクリプト共通のデフォルトパス。必要なら CLI オプションで差し替え。
DEFAULT_DATA = ROOT_DIR / "data" / "Titanic-Dataset.csv"
DEFAULT_TEST_DATA = ROOT_DIR / "data" / "Titanic-Dataset.csv"
DEFAULT_MODEL_PATH = ROOT_DIR / "models" / "titanic" / "lightgbm_pipeline.pkl"
DEFAULT_REPORT_DIR = ROOT_DIR / "reports" / "titanic" / "lightgbm"

TARGET = "Survived"
FeatureKind = Literal["numeric", "categorical"]
IGNORED_COLUMNS = {"PassengerId", "Name", "Ticket", "Cabin"}


@dataclass(frozen=True)
class FeatureSpec:
    """特徴量名と型の種別を保持するヘルパー。"""

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


def load_dataset(path: Path, feature_specs: Optional[list[FeatureSpec]] = None) -> tuple[pd.DataFrame, pd.Series, list[FeatureSpec]]:
    df = pd.read_csv(path)
    if TARGET not in df.columns:
        raise ValueError(f"{path} is missing target column '{TARGET}'.")

    df = df.dropna(subset=[TARGET])  # ラベル欠損行は除外しておく
    df = df.drop(columns=[col for col in IGNORED_COLUMNS if col in df.columns], errors="ignore")

    specs = feature_specs or infer_feature_specs(df)
    feature_names = [spec.name for spec in specs]
    missing_cols = set(feature_names) - set(df.columns)
    if missing_cols:
        raise ValueError(f"{path} is missing required features: {missing_cols}")

    X = df[feature_names].copy()
    y = df[TARGET].astype(int)

    numeric_features = split_feature_names(specs, "numeric")
    categorical_features = split_feature_names(specs, "categorical")

    for col in numeric_features:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    for col in categorical_features:
        X[col] = X[col].astype("string").fillna("")

    return X, y, specs


def build_pipeline(random_state: int, params: dict, feature_specs: list[FeatureSpec]) -> Pipeline:
    numeric_features = split_feature_names(feature_specs, "numeric")
    categorical_features = split_feature_names(feature_specs, "categorical")

    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = lgb.LGBMClassifier(
        n_estimators=params["n_estimators"],
        learning_rate=params["learning_rate"],
        num_leaves=params["num_leaves"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        min_child_samples=params["min_child_samples"],
        reg_alpha=params["reg_alpha"],
        reg_lambda=params["reg_lambda"],
        objective="binary",
        random_state=random_state,
        n_jobs=-1,
    )

    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def sanitize_label(label: str) -> str:
    return label.lower().replace(" ", "_")


def evaluate(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    label: str,
    output_dir: Path,
) -> None:
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    report = classification_report(y_test, preds)
    roc_auc = roc_auc_score(y_test, proba)
    acc = accuracy_score(y_test, preds)
    fpr, tpr, _ = roc_curve(y_test, proba)

    print(f"\n=== Evaluation on {label} set ===")
    print(report)
    print(f"ROC AUC: {roc_auc:.4f}")

    output_dir.mkdir(parents=True, exist_ok=True)
    label_key = sanitize_label(label)

    report_path = output_dir / f"{label_key}_classification_report.txt"
    with report_path.open("w") as f:
        f.write(report)

    metrics = {
        "roc_auc": roc_auc,
        "accuracy": acc,
        "support": len(y_test),
    }
    metrics_path = output_dir / f"{label_key}_metrics.json"
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)

    # ROC 曲線を画像として保存しておく
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {label}")
    plt.legend(loc="lower right")
    roc_path = output_dir / f"{label_key}_roc_curve.png"
    plt.tight_layout()
    plt.savefig(roc_path, dpi=120)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LightGBM on Titanic dataset.")
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA,
        help="Path to the Titanic CSV file.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Where to store the trained model pipeline.",
    )
    parser.add_argument(
        "--test-data",
        type=Path,
        default=DEFAULT_TEST_DATA,
        help="External dataset used only for final evaluation.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data for validation/test split.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=1,
        help="Number of Optuna trials for hyperparameter search.",
    )
    parser.add_argument(
        "--tune-sample-size",
        type=int,
        default=200_000,
        help="Number of rows to use during hyperparameter tuning (use entire dataset if smaller or if set to 0).",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help="Directory to store evaluation reports and plots.",
    )
    return parser.parse_args()


def tune_hyperparameters(
    X: pd.DataFrame,
    y: pd.Series,
    feature_specs: list[FeatureSpec],
    test_size: float,
    random_state: int,
    n_trials: int,
    sample_size: int,
) -> dict:
    if sample_size > 0 and len(X) > sample_size:
        X, _, y, _ = train_test_split(
            X,
            y,
            train_size=sample_size,
            stratify=y,
            random_state=random_state,
        )

    def objective(trial: optuna.trial.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 64),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 1.0, log=True),
        }

        pipeline = build_pipeline(random_state, params, feature_specs)
        X_train, X_valid, y_train, y_valid = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )
        pipeline.fit(X_train, y_train)
        proba = pipeline.predict_proba(X_valid)[:, 1]
        return roc_auc_score(y_valid, proba)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"Best trial ROC AUC: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    return study.best_params


def main() -> None:
    args = parse_args()
    X_train_full, y_train_full, feature_specs = load_dataset(args.data)
    X_external, y_external, _ = load_dataset(args.test_data, feature_specs=feature_specs)

    best_params = tune_hyperparameters(
        X_train_full,
        y_train_full,
        feature_specs,
        args.test_size,
        args.random_state,
        args.n_trials,
        args.tune_sample_size,
    )

    pipeline = build_pipeline(args.random_state, best_params, feature_specs)
    pipeline.fit(X_train_full, y_train_full)
    evaluate(
        pipeline,
        X_external,
        y_external,
        label="external test",
        output_dir=args.report_dir,
    )

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, args.model_path)
    print(f"Saved LightGBM pipeline to {args.model_path}")


if __name__ == "__main__":
    main()
