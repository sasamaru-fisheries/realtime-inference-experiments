# Titanic Survival Pipeline – Python & Java

Titanic データセットを素材に、Python でモデルを学習して ONNX / PMML アーティファクトを出力し、Java から PMML を読み込んで推論するまでを一通りまとめています。

---

## ディレクトリ構成

```
.
├── data/                         # 入力データ (Titanic-Dataset.csv)
├── model/                        # エクスポート済みモデル (ONNX / PMML)
├── models/
│   └── titanic/                  # Python 側で再利用するパイプライン pickle
├── reports/                      # 評価レポート (各トレーニングスクリプトが生成)
├── src/
│   ├── train_random_forest.py    # RandomForest パイプラインの学習 + 評価 + レポート
│   ├── train_lightgbm.py         # LightGBM パイプラインの学習 + 評価 + レポート
│   ├── export_to_pmml.py         # RandomForest pickle を PMML へ変換
│   └── sample_batch.txt          # Java CLI 用の推論サンプル
├── train.py                      # 両モデルを一括学習し ONNX + pickle を保存
├── pmml-predictor/               # PMML を読み込む Java CLI
└── README.md
```

---

## 1. 環境セットアップ

推奨: [uv](https://github.com/astral-sh/uv) で仮想環境を構築します。

```bash
uv sync
source .venv/bin/activate
```

もしくは従来の pip を使って:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install -e .
```

Python 3.12 以降を想定しています。

---

## 2. モデルの学習とエクスポート

### (A) まとめて実行する場合

`train.py` を実行すると、Titanic データに対して以下を自動で行います。

- RandomForest / LightGBM の前処理付きパイプラインを学習
- ONNX 形式で `model/titanic_random_forest.onnx` / `model/titanic_lightgbm.onnx` を出力
- Python 再利用用に `models/titanic/random_forest_pipeline.pkl` / `models/titanic/lightgbm_pipeline.pkl` を保存
- ONNX Runtime を使って 5 サンプルの動作確認を実行（標準出力に予測を表示）

```bash
python train.py
```

生成物のサマリ:

| ファイル | 用途 |
| --- | --- |
| `model/titanic_random_forest.onnx` | RandomForest パイプライン (標準スカラー + OneHot + RF) |
| `model/titanic_lightgbm.onnx` | LightGBM パイプライン（ONNX 内で前処理＋軽量GBDTを一体化） |
| `models/titanic/random_forest_pipeline.pkl` | sklearn パイプライン丸ごとの pickle |
| `models/titanic/lightgbm_pipeline.pkl` | LightGBM パイプライン丸ごとの pickle |

### (B) 個別に操作したい場合

より詳細な制御や評価レポートが必要な場合は `src/` 以下を利用します。

1. RandomForest を学習（Optuna で簡易チューニング、ROC 曲線やメトリクスを保存）
   ```bash
   python src/train_random_forest.py \
     --data data/Titanic-Dataset.csv \
     --test-data data/Titanic-Dataset.csv \
     --report-dir reports/titanic/random_forest
   ```
2. LightGBM を学習
   ```bash
   python src/train_lightgbm.py \
     --data data/Titanic-Dataset.csv \
     --test-data data/Titanic-Dataset.csv \
     --report-dir reports/titanic/lightgbm
   ```
3. RandomForest パイプラインを PMML に変換（Java 連携用）
   ```bash
   python src/export_to_pmml.py
   ```

---

## 3. ONNX モデルの検証

`train.py` 実行時に onnxruntime を用いた推論検証を行っています。既存の ONNX を確認したい場合は Python から直接呼び出してください。

```python
import numpy as np
import onnxruntime as ort

session = ort.InferenceSession("model/titanic_random_forest.onnx")
sample = {
    "Pclass": np.array([[3]], dtype=object),
    "Sex": np.array([["male"]], dtype=object),
    "Age": np.array([[22.0]], dtype=np.float32),
    "SibSp": np.array([[1.0]], dtype=np.float32),
    "Parch": np.array([[0.0]], dtype=np.float32),
    "Fare": np.array([[7.25]], dtype=np.float32),
    "Embarked": np.array([["S"]], dtype=object),
}
prob = session.run(None, sample)
print(prob)
```

---

## 4. PMML への変換

`src/export_to_pmml.py` は、学習済み RandomForest パイプライン（pickle）を読み込み、sklearn2pmml を使って `model/titanic_random_forest.pmml` を出力します。Java CLI はこの PMML を参照します。

生成手順:

```bash
python src/train_random_forest.py    # 未実行なら先に学習
python src/export_to_pmml.py         # model/titanic_random_forest.pmml が作成される
```

---

## 5. Java (PMML) 推論手順

1. **ビルド**
   ```bash
   cd pmml-predictor
   mvn -q clean package
   cd ..
   ```
   `pmml-predictor/target/pmml-predictor-1.0-SNAPSHOT.jar` が生成されます。

2. **バッチ入力を用意**  
   `src/sample_batch.txt` に複数行の乗客データ（`Pclass Sex Age SibSp Parch Fare Embarked`）が入っています。単発推論なら CLI の末尾に直接入力できます。

3. **推論実行**
   ```bash
   java -jar pmml-predictor/target/pmml-predictor-1.0-SNAPSHOT.jar \
     --model model/titanic_random_forest.pmml \
     --batch src/sample_batch.txt
   ```

4. **ホットリロード (任意)**
   ```bash
   java -jar pmml-predictor/target/pmml-predictor-1.0-SNAPSHOT.jar --watch
   ```
   `model/titanic_random_forest.pmml` が更新されると自動で再読み込みします。

---

## 6. よくある質問

- **モデルを作り直したい:** `train.py` または `src/train_random_forest.py` / `src/train_lightgbm.py` を再実行してください。PMML も更新するなら `src/export_to_pmml.py` を合わせて再実行します。
- **パスを変えたい:** 各スクリプトには `--data`, `--model-path`, `--report-dir` などのオプションがあります。コマンドラインで指定すればデフォルト以外のディレクトリを利用できます。
- **LightGBM を Java から使いたい:** 現状 Java 側は PMML の RandomForest を前提にしています。LightGBM を Java で使うには ONNX Runtime など別途構築が必要です。

---

## ライセンス

教育目的のサンプルです。必要に応じて適切なライセンス表記を追加してください。
