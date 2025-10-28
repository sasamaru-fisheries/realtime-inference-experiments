"""Titanic データで RandomForest / LightGBM を学習し、ONNX と pickle の両方を保存する。"""

from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd
import onnx
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType, StringTensorType
from onnxmltools import convert_lightgbm
from onnxmltools.convert.common.data_types import FloatTensorType as LightGBMFloatTensorType

TARGET_OPSET = 15  # onnxruntime で安定して動く opset
PICKLE_SUBDIR = "titanic"
FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
NUMERIC_FEATURES = ["Age", "SibSp", "Parch", "Fare"]
CATEGORICAL_FEATURES = ["Pclass", "Sex", "Embarked"]
DATA_FILENAME = "Titanic-Dataset.csv"
TARGET = "Survived"


def load_dataset(data_path: Path, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Titanic データを読み込み、学習/テスト用に分割する。"""
    df = pd.read_csv(data_path)
    missing_cols = set(FEATURES + [TARGET]) - set(df.columns)
    if missing_cols:
        raise ValueError(f"必要な列が不足しています: {missing_cols}")

    df = df.dropna(subset=[TARGET])  # 目的変数欠損は学習に使えないので除外

    X = df[FEATURES].copy()
    y = df[TARGET].astype(int)

    for col in NUMERIC_FEATURES:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    for col in CATEGORICAL_FEATURES:
        original = X[col]
        values = original.astype(str)
        values[original.isna()] = ""
        X[col] = values

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def build_preprocessor() -> ColumnTransformer:
    """数値・カテゴリ列への前処理を定義する。"""
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(missing_values="", strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, NUMERIC_FEATURES),
            ("categorical", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )


def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42) -> Pipeline:
    """前処理込みの RandomForest パイプラインを学習する。"""
    pipeline = Pipeline([
        ("preprocess", build_preprocessor()),
        ("classifier", RandomForestClassifier(n_estimators=300, random_state=random_state)),
    ])
    pipeline.fit(X_train, y_train)
    return pipeline


def train_lightgbm(X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42) -> Pipeline:
    """前処理込みの LightGBM パイプラインを学習する。"""
    pipeline = Pipeline([
        ("preprocess", build_preprocessor()),
        ("classifier", LGBMClassifier(
            objective="binary",
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            random_state=random_state,
            verbose=-1,
        )),
    ])
    pipeline.fit(X_train, y_train)
    return pipeline


def get_initial_types() -> list[tuple[str, FloatTensorType | StringTensorType]]:
    """skl2onnx へ渡す入力スキーマを組み立てる。"""
    return [
        ("Pclass", StringTensorType([None, 1])),
        ("Sex", StringTensorType([None, 1])),
        ("Age", FloatTensorType([None, 1])),
        ("SibSp", FloatTensorType([None, 1])),
        ("Parch", FloatTensorType([None, 1])),
        ("Fare", FloatTensorType([None, 1])),
        ("Embarked", StringTensorType([None, 1])),
    ]


def export_pipeline_to_onnx(pipeline: Pipeline, output_path: Path) -> None:
    """sklearn パイプラインを ONNX に変換して保存する。"""
    classifier = pipeline.named_steps["classifier"]
    if isinstance(classifier, LGBMClassifier):
        preprocessor = pipeline.named_steps["preprocess"]
        preprocess_onnx = convert_sklearn(
            preprocessor,
            initial_types=get_initial_types(),
            target_opset=TARGET_OPSET,
        )
        feature_count = classifier.booster_.num_feature()
        lightgbm_onnx = convert_lightgbm(
            classifier.booster_,
            initial_types=[("preprocessed", LightGBMFloatTensorType([None, feature_count]))],
            target_opset=TARGET_OPSET,
            zipmap=False,
        )
        # LightGBM 側の IR / opset を前処理 ONNX に揃えてからマージする
        lightgbm_onnx.ir_version = preprocess_onnx.ir_version
        domain_to_version = {op.domain: op.version for op in lightgbm_onnx.opset_import}
        for op in preprocess_onnx.opset_import:
            if op.domain in domain_to_version:
                domain_to_version[op.domain] = max(domain_to_version[op.domain], op.version)
            else:
                domain_to_version[op.domain] = op.version
        del lightgbm_onnx.opset_import[:]
        for domain, version in domain_to_version.items():
            lightgbm_onnx.opset_import.append(
                onnx.helper.make_operatorsetid(domain, version)
            )
        merged_model = onnx.compose.merge_models(
            preprocess_onnx,
            lightgbm_onnx,
            io_map=[(preprocess_onnx.graph.output[0].name, lightgbm_onnx.graph.input[0].name)],
        )
        onnx_model = merged_model
    else:
        options = {id(classifier): {"zipmap": False}}
        onnx_model = convert_sklearn(
            pipeline,
            initial_types=get_initial_types(),
            target_opset=TARGET_OPSET,
            options=options,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(onnx_model.SerializeToString())
    print(f"ONNX モデルを書き出しました: {output_path}")


def run_onnx_inference(model_path: Path, inputs: pd.DataFrame) -> dict[str, np.ndarray]:
    """ONNX モデルを実行し、出力を辞書で返す。"""
    session = ort.InferenceSession(model_path.as_posix(), providers=["CPUExecutionProvider"])
    feed = {}
    for feature in ["Age", "SibSp", "Parch", "Fare"]:
        feed[feature] = inputs[[feature]].to_numpy(dtype=np.float32)
    for feature in ["Pclass", "Sex", "Embarked"]:
        feed[feature] = inputs[[feature]].to_numpy(dtype=object)  # onnxruntime は str を object dtype で受け取る
    outputs = session.get_outputs()
    results = session.run(None, feed)
    return {output.name: value for output, value in zip(outputs, results)}


def decode_classification_outputs(output_map: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray | None]:
    """ONNX 出力から予測ラベルと確率を抽出する。"""
    for name in ("output_probability", "probabilities", "probability"):
        if name in output_map:
            probabilities = np.asarray(output_map[name])
            return probabilities.argmax(axis=1), probabilities

    for name in ("output_label", "label"):
        if name in output_map:
            labels = np.asarray(output_map[name])
            return labels.astype(int), None

    first_value = next(iter(output_map.values()))
    if first_value.ndim == 2:
        return first_value.argmax(axis=1), first_value
    return first_value.astype(int), None


def main() -> None:
    project_root = Path(__file__).resolve().parent
    data_path = project_root / "data" / DATA_FILENAME
    onnx_dir = project_root / "model"
    pickle_dir = project_root / "models" / PICKLE_SUBDIR

    X_train, X_test, y_train, y_test = load_dataset(data_path)

    rf_pipeline = train_random_forest(X_train, y_train)
    rf_onnx_path = onnx_dir / "titanic_random_forest.onnx"
    export_pipeline_to_onnx(rf_pipeline, rf_onnx_path)
    rf_pickle_path = pickle_dir / "random_forest_pipeline.pkl"
    rf_pickle_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf_pipeline, rf_pickle_path)
    print(f"RandomForest パイプラインを保存しました: {rf_pickle_path}")

    lgbm_pipeline = train_lightgbm(X_train, y_train)
    lgbm_onnx_path = onnx_dir / "titanic_lightgbm.onnx"
    export_pipeline_to_onnx(lgbm_pipeline, lgbm_onnx_path)
    lgbm_pickle_path = pickle_dir / "lightgbm_pipeline.pkl"
    joblib.dump(lgbm_pipeline, lgbm_pickle_path)
    print(f"LightGBM パイプラインを保存しました: {lgbm_pickle_path}")

    sample_inputs = X_test.head(5).copy()

    rf_outputs = run_onnx_inference(rf_onnx_path, sample_inputs)
    rf_predictions, _ = decode_classification_outputs(rf_outputs)
    print("RandomForest ONNX predictions:", rf_predictions.tolist())

    lgbm_outputs = run_onnx_inference(lgbm_onnx_path, sample_inputs)
    lgbm_predictions, _ = decode_classification_outputs(lgbm_outputs)
    print("LightGBM ONNX predictions:", lgbm_predictions.tolist())


if __name__ == "__main__":
    main()
