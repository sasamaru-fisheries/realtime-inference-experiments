"""Titanic データで RandomForest / LightGBM を学習し、ONNX と pickle の両方を保存する。"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Tuple

import joblib
import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
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
DATA_FILENAME = "Titanic-Dataset.csv"
TARGET = "Survived"
IGNORED_COLUMNS = {"PassengerId", "Name", "Ticket", "Cabin"}  # ID など学習で使わない列

FeatureKind = Literal["numeric", "categorical"]


@dataclass(frozen=True)
class FeatureSpec:
    """データセット内の特徴量と型の種別を保持する。"""

    name: str
    kind: FeatureKind


def infer_feature_specs(df: pd.DataFrame, target: str) -> list[FeatureSpec]:
    """DataFrame から目的変数以外の列を調べ、数値/カテゴリの種別に仕分けする。"""
    specs: list[FeatureSpec] = []
    for column in df.columns:
        if column == target:
            continue

        series = df[column]
        if pd.api.types.is_numeric_dtype(series):
            specs.append(FeatureSpec(column, "numeric"))
            continue

        # 文字列カラムでも数値に変換できる場合は数値扱いにする
        numeric_candidate = pd.to_numeric(series, errors="coerce")
        if numeric_candidate.notna().mean() >= 0.95:
            specs.append(FeatureSpec(column, "numeric"))
        else:
            specs.append(FeatureSpec(column, "categorical"))
    return specs


def split_feature_names(feature_specs: list[FeatureSpec], kind: FeatureKind) -> list[str]:
    """FeatureSpec から指定された種別の列名だけを抽出する。"""
    return [spec.name for spec in feature_specs if spec.kind == kind]


def load_dataset(
    data_path: Path,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list[FeatureSpec]]:
    """Titanic データを読み込み、特徴量の型を推定したうえで学習/テスト用に分割する。"""
    df = pd.read_csv(data_path)
    if TARGET not in df.columns:
        raise ValueError(f"{data_path} に目的変数 '{TARGET}' が存在しません。")

    df = df.dropna(subset=[TARGET])  # 目的変数欠損は学習に使えないので除外
    df = df.drop(columns=[col for col in IGNORED_COLUMNS if col in df.columns], errors="ignore")
    feature_specs = infer_feature_specs(df, TARGET)
    feature_names = [spec.name for spec in feature_specs]

    missing_cols = set(feature_names) - set(df.columns)
    if missing_cols:
        raise ValueError(f"{data_path} に必要な列が見つかりません: {missing_cols}")

    X = df[feature_names].copy()
    y = df[TARGET].astype(int)

    numeric_features = split_feature_names(feature_specs, "numeric")
    categorical_features = split_feature_names(feature_specs, "categorical")

    for col in numeric_features:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    for col in categorical_features:
        values = X[col].astype("string")
        X[col] = values.fillna("")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test, feature_specs


def build_preprocessor(feature_specs: list[FeatureSpec]) -> ColumnTransformer:
    """数値・カテゴリ列への前処理を定義する。"""
    numeric_features = split_feature_names(feature_specs, "numeric")
    categorical_features = split_feature_names(feature_specs, "categorical")

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(missing_values="", strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("categorical", categorical_transformer, categorical_features),
        ]
    )


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_specs: list[FeatureSpec],
    random_state: int = 42,
) -> Pipeline:
    """前処理込みの RandomForest パイプラインを学習する。"""
    pipeline = Pipeline([
        ("preprocess", build_preprocessor(feature_specs)),
        ("classifier", RandomForestClassifier(n_estimators=300, random_state=random_state)),
    ])
    pipeline.fit(X_train, y_train)
    return pipeline


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_specs: list[FeatureSpec],
    random_state: int = 42,
) -> Pipeline:
    """前処理込みの LightGBM パイプラインを学習する。"""
    pipeline = Pipeline([
        ("preprocess", build_preprocessor(feature_specs)),
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


def get_initial_types(feature_specs: list[FeatureSpec]) -> list[tuple[str, FloatTensorType | StringTensorType]]:
    """skl2onnx へ渡す入力スキーマを FeatureSpec から組み立てる。"""
    initial_types: list[tuple[str, FloatTensorType | StringTensorType]] = []
    for spec in feature_specs:
        if spec.kind == "numeric":
            initial_types.append((spec.name, FloatTensorType([None, 1])))
        else:
            initial_types.append((spec.name, StringTensorType([None, 1])))
    return initial_types


def export_pipeline_to_onnx(
    pipeline: Pipeline,
    feature_specs: list[FeatureSpec],
    output_path: Path,
) -> None:
    """sklearn パイプラインを ONNX に変換して保存する。"""
    classifier = pipeline.named_steps["classifier"]
    if isinstance(classifier, LGBMClassifier):
        preprocessor = pipeline.named_steps["preprocess"]
        preprocess_onnx = convert_sklearn(
            preprocessor,
            initial_types=get_initial_types(feature_specs),
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
            initial_types=get_initial_types(feature_specs),
            target_opset=TARGET_OPSET,
            options=options,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(onnx_model.SerializeToString())
    print(f"ONNX モデルを書き出しました: {output_path}")


def run_onnx_inference(
    model_path: Path,
    inputs: pd.DataFrame,
    feature_specs: list[FeatureSpec],
) -> dict[str, np.ndarray]:
    """ONNX モデルを実行し、出力を辞書で返す。"""
    session = ort.InferenceSession(model_path.as_posix(), providers=["CPUExecutionProvider"])
    feed: dict[str, np.ndarray] = {}
    for spec in feature_specs:
        column = inputs[[spec.name]]
        if spec.kind == "numeric":
            feed[spec.name] = column.to_numpy(dtype=np.float32)
        else:
            feed[spec.name] = column.astype("string").fillna("").to_numpy(dtype=object)  # onnxruntime は str を object dtype で受け取る
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

    X_train, X_test, y_train, y_test, feature_specs = load_dataset(data_path)

    rf_pipeline = train_random_forest(X_train, y_train, feature_specs)
    rf_onnx_path = onnx_dir / "titanic_random_forest.onnx"
    export_pipeline_to_onnx(rf_pipeline, feature_specs, rf_onnx_path)
    rf_pickle_path = pickle_dir / "random_forest_pipeline.pkl"
    rf_pickle_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf_pipeline, rf_pickle_path)
    print(f"RandomForest パイプラインを保存しました: {rf_pickle_path}")

    lgbm_pipeline = train_lightgbm(X_train, y_train, feature_specs)
    lgbm_onnx_path = onnx_dir / "titanic_lightgbm.onnx"
    export_pipeline_to_onnx(lgbm_pipeline, feature_specs, lgbm_onnx_path)
    lgbm_pickle_path = pickle_dir / "lightgbm_pipeline.pkl"
    joblib.dump(lgbm_pipeline, lgbm_pickle_path)
    print(f"LightGBM パイプラインを保存しました: {lgbm_pickle_path}")

    sample_inputs = X_test.head(5).copy()

    rf_outputs = run_onnx_inference(rf_onnx_path, sample_inputs, feature_specs)
    rf_predictions, _ = decode_classification_outputs(rf_outputs)
    print("RandomForest ONNX predictions:", rf_predictions.tolist())

    lgbm_outputs = run_onnx_inference(lgbm_onnx_path, sample_inputs, feature_specs)
    lgbm_predictions, _ = decode_classification_outputs(lgbm_outputs)
    print("LightGBM ONNX predictions:", lgbm_predictions.tolist())


if __name__ == "__main__":
    main()
