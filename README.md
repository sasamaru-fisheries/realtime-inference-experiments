# Titanic Survival Prediction – Python & Java

This repository trains Titanic survival models in Python and serves them from Java.  
The typical workflow is:

1. train and evaluate the scikit-learn pipelines on Titanic data;  
2. export the RandomForest pipeline to PMML for cross-language inference;  
3. run the Java CLI to score new passengers with the exported model.

Only the Titanic pipeline is maintained (the former Iris sample has been removed).

---

## Repository Layout

```
.
├── data/                         # Input CSV(s). Default: Titanic-Dataset.csv
├── model/                        # Exported artifacts (eg. titanic_random_forest.pmml)
├── models/
│   └── titanic/                  # Pickled Python pipelines (RandomForest / LightGBM)
├── reports/
│   └── titanic/                  # Evaluation reports & ROC curves generated at training time
├── titanic/
│   ├── train_random_forest.py    # Python training + Optuna tuning + evaluation (RandomForest)
│   ├── train_lightgbm.py         # Python training + evaluation (LightGBM)
│   ├── export_to_pmml.py         # Convert the trained RandomForest pipeline into PMML
│   └── sample_batch.txt          # Example batch input for Java inference
├── pmml-predictor/               # Maven CLI (fat JAR) for PMML inference
├── standalone-pmml/              # PMML inference without Maven (pre-bundled libs)
└── README.md
```

---

## Python Training & Evaluation

1. **Install dependencies**

   ```bash
   python3 -m pip install pandas scikit-learn optuna lightgbm sklearn2pmml matplotlib
   ```

2. **Train the RandomForest pipeline**

   ```bash
   python titanic/train_random_forest.py \
     --data data/Titanic-Dataset.csv \
     --test-data data/Titanic-Dataset.csv \
     --tune-sample-size 0 \
     --report-dir reports/titanic/random_forest
   ```

   - Performs a small Optuna search (1 trial by default) and fits the final model.  
   - Saves the trained pipeline to `models/titanic/random_forest_pipeline.pkl`.  
   - Writes evaluation artefacts into `reports/titanic/random_forest/`:
     - `external_test_classification_report.txt`
     - `external_test_metrics.json` (accuracy / ROC AUC / support)
     - `external_test_roc_curve.png`

3. **Train the LightGBM pipeline** (optional)

   ```bash
   python titanic/train_lightgbm.py \
     --data data/Titanic-Dataset.csv \
     --test-data data/Titanic-Dataset.csv \
     --tune-sample-size 0 \
     --report-dir reports/titanic/lightgbm
   ```

   This script stores `models/titanic/lightgbm_pipeline.pkl` and produces matching reports under `reports/titanic/lightgbm/`.

4. **Export RandomForest to PMML (for Java)**

   ```bash
   python titanic/export_to_pmml.py
   ```

   The script reloads the pickled pipeline, refits it on the full dataset for PMML compatibility, and exports to `model/titanic_random_forest.pmml`.  
   (LightGBM は ONNX 変換が必要になるため、Java では PMML 版 RandomForest を利用するのが簡単です。)

---

## Java 推論ワークフロー（PMML）

1. **Maven プロジェクトをビルド**

   ```bash
   cd pmml-predictor
   mvn -q clean package
   cd ..
   ```

   生成物: `pmml-predictor/target/pmml-predictor-1.0-SNAPSHOT.jar`

2. **推論用の入力を用意**

   `pmml-predictor` CLI は 7 特徴（数値 + 文字列）を Titanic の順番で受け取ります：

   ```text
   Pclass Sex Age SibSp Parch Fare Embarked
   ```

   例 (空白区切りでもカンマ区切りでも可):

   ```text
   3 male 22 1 0 7.25 S
   1 female 38 1 0 71.2833 C
   ```

   複数件を一括で推論する場合は上記形式の行を `titanic/sample_batch.txt` のようにファイルへ記述します。

3. **PMML モデルで推論を実行**

   ```bash
   java -jar pmml-predictor/target/pmml-predictor-1.0-SNAPSHOT.jar \
     --model model/titanic_random_forest.pmml \
     --batch titanic/sample_batch.txt
   ```

   `--model` を省略すると `model/titanic_random_forest.pmml` が自動参照されます。  
   `--batch` を省略すればコマンドライン引数で単発推論も可能です（例: `... 3 male 22 1 0 7.25 S`）。

   出力例:

   ```text
   Model loaded from: ...

   === Sample 1 ===
   Input features (Titanic order: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked):
     Pclass    = 3.0000
     Sex       = male
     Age       = 22.0000
     SibSp     = 1.0000
     Parch     = 0.0000
     Fare      = 7.2500
     Embarked  = S

   Predicted class id: 0
   Predicted class label: not_survived

   Class probabilities:
     probability(0)   : 0.8819
     probability(1)   : 0.1181
   ```

4. **ホットリロード (watch モード)**

   モデルファイルが更新されたら自動で読み直す場合は `--watch` を付けて起動します。

   ```bash
   java -jar pmml-predictor/target/pmml-predictor-1.0-SNAPSHOT.jar \
     --watch \
     --batch titanic/sample_batch.txt
   ```

   - 起動直後にバッチの内容を評価し、その後はコンソールにプロンプトが表示されます。  
     `Pclass Sex Age SibSp Parch Fare Embarked` を空白またはカンマで区切って入力すると、その場で推論されます。  
     `:exit` を入力すると終了します。
   - `model/titanic_random_forest.pmml` が変更されると自動的に再ロードを試みます。  
     成功時は「Model reload succeeded」、失敗時はスタックトレース付きで警告が出ます。失敗しても直前のモデルでサービスを継続します。

   **ホットリロード検証手順の例**

   1. 上記コマンドで watch モードを起動し、別ターミナルで以下を実行:
      ```bash
      python titanic/train_random_forest.py --data data/Titanic-Dataset.csv --test-data data/Titanic-Dataset.csv
      python titanic/export_to_pmml.py
      ```
   2. Java 側のコンソールに `Detected change...` → `Model reload succeeded.` が出ればリロード完了。  
   3. コンソールに新しいサンプルを入力し、更新済みモデルの推論結果を確認します。

   モデルファイルを差し替えるだけでロックレスに更新されるため、再コンパイル・再起動は不要です。

---

## Maven を使わずに推論したい場合

`standalone-pmml/` には依存ライブラリを含んだ構成を用意しています。

```bash
cd standalone-pmml
javac -cp "libs/*" PMMLPredictor.java
java -cp ".:libs/*" PMMLPredictor --model ../model/titanic_random_forest.pmml --batch ../titanic/sample_batch.txt
```

（ONNX 版が必要な場合は `standalone-onnx/` を同様に利用できます。）

---

## Q&A / Tips

- **モデルを再学習したら？**  
  `titanic/train_random_forest.py` を再実行し、新しい `models/titanic/random_forest_pipeline.pkl` を生成してから `titanic/export_to_pmml.py` を流せば PMML を更新できます。Java 側の再ビルドは不要です。

- **データセットの場所を変えたい**  
  すべてのスクリプトに `--data` / `--test-data` / `--model-path` / `--report-dir` オプションがあります。パスを変更する場合は各オプションを指定してください。

- **ホットリロード時のエラー処理は？**  
  `--watch` モードでは新しい PMML の読み込みに失敗するとスタックトレース付きで警告を出し、直前のモデルを保持したまま処理を継続します。ログを確認しつつ問題を修正し、再度 PMML を上書きしてください。

- **LightGBM を Java で使いたい**  
  LightGBM パイプラインは pickle (`models/titanic/lightgbm_pipeline.pkl`) に保存されているので、Python で直接利用できます。Java から使いたい場合は別途 ONNX 変換が必要です（本リポジトリでは PMML 版 RandomForest のみサポート）。

---

## ライセンス

このサンプルは教育目的で提供されています。必要に応じてライセンス表記を追加してご利用ください。
