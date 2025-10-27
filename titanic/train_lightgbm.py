import argparse
import json
from pathlib import Path

import joblib
import optuna
import lightgbm as lgb
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
DEFAULT_DATA = ROOT_DIR / "data" / "Titanic-Dataset.csv"
DEFAULT_TEST_DATA = ROOT_DIR / "data" / "Titanic-Dataset.csv"
DEFAULT_MODEL_PATH = ROOT_DIR / "models" / "titanic" / "lightgbm_pipeline.pkl"
DEFAULT_REPORT_DIR = ROOT_DIR / "reports" / "titanic" / "lightgbm"


FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
TARGET = "Survived"


def build_pipeline(random_state: int, params: dict) -> Pipeline:
    numeric_features = ["Age", "SibSp", "Parch", "Fare"]
    categorical_features = ["Pclass", "Sex", "Embarked"]

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


def load_dataset(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    missing_cols = set(FEATURES + [TARGET]) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Dataset {path} is missing required columns: {missing_cols}")

    df = df.dropna(subset=[TARGET])
    X = df[FEATURES].copy()
    y = df[TARGET].astype(int)
    return X, y


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
            "n_estimators": trial.suggest_int("n_estimators", 300, 1200),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 256),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 200),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

        pipeline = build_pipeline(random_state, params)
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
    X_train_full, y_train_full = load_dataset(args.data)
    X_external, y_external = load_dataset(args.test_data)

    best_params = tune_hyperparameters(
        X_train_full,
        y_train_full,
        args.test_size,
        args.random_state,
        args.n_trials,
        args.tune_sample_size,
    )

    pipeline = build_pipeline(args.random_state, best_params)
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
