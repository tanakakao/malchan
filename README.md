# malchan

`malchan` は、材料・化学データを想定した機械学習ワークフローを試作するための Python パッケージです。回帰・分類モデルの学習、モデル比較、特徴量の前処理、説明可能性の可視化、Optuna を使った逆解析、Excel 形式での結果出力をまとめて扱うことを目指しています。

このプロジェクトは開発初期段階です。現時点では後方互換性よりも、パッケージとして import できること、公開 API の入口を整理すること、旧ディレクトリ構成由来の import を解消することを優先しています。

---

## 主な機能

- scikit-learn 互換の機械学習パイプライン作成
- 回帰・分類モデルの学習、予測、交差検証
- 学習済みモデルからの候補モデル比較と最良モデル選定
- Optuna によるハイパーパラメータチューニング
- SHAP、Permutation Feature Importance、Partial Dependence などの可視化補助
- SMILES や組成式を使った特徴量生成の補助
- 学習済みモデルから直接実行できる逆解析・候補探索
- 学習結果や図表の Excel 出力

---

## パッケージ構成

```text
src/malchan/
├── app/                 # FastAPI / Web アプリ用のアプリケーション層
├── pipeline/            # 単一・複数目的の学習済みモデル本体
├── models/              # モデル生成、比較、前処理、説明可能性
├── visualization/       # Plotly ベースの可視化
├── inverse_analysis/    # Optuna による逆解析
└── export/              # Excel 出力関連
```

主な入口は以下です。

| Module | Purpose |
|---|---|
| `malchan.pipeline` | 学習、予測、比較、逆解析を行うモデル本体 |
| `malchan.app` | FastAPI アプリファクトリ、設定、将来の Web UI 用アプリケーション層 |
| `malchan.models` | モデルパイプライン、比較結果型、前処理、説明可能性の処理 |
| `malchan.visualization` | モデル結果・逆解析結果の可視化 |
| `malchan.inverse_analysis` | Optuna ベースの低水準な逆解析関数 |
| `malchan.export` | Excel レポート出力 |

---

## インストール

ローカル開発用の最小構成です。

```bash
git clone <repository-url>
cd malchan
pip install -e .
```

モデル作成、材料特徴量、可視化、逆解析、Excel 出力をまとめて使う場合は optional dependencies を追加します。

```bash
pip install -e ".[models,materials,visualization,inverse,export]"
```

開発用ツールを含める場合:

```bash
pip install -e ".[dev,models,materials,visualization,inverse,export,web]"
```

すべての optional dependencies を入れる場合:

```bash
pip install -e ".[all]"
```

---

## モデルの学習と予測

```python
import pandas as pd

from malchan.models import SingleOutputMLModelPipeline


df = pd.DataFrame(
    {
        "x1": [0.1, 0.2, 0.3, 0.4],
        "x2": [1.0, 0.9, 0.7, 0.4],
        "y": [10.0, 12.0, 13.5, 16.0],
    }
)

model = SingleOutputMLModelPipeline()
model.fit(
    df=df,
    target_col="y",
    task="regression",
    num_cols=["x1", "x2"],
    cat_cols=[],
    model_names=["ランダムフォレスト回帰"],
)

pred = model.predict(df[["x1", "x2"]])
print(pred)
```

## 学習済みモデルから候補モデルを比較

`compare()` は、現在のモデルが保持する学習データ、列定義、材料特徴量、前処理設定を再利用します。候補ごとに同じ交差検証条件で学習・評価し、回帰では既定でテスト RMSE の昇順、分類ではテスト F1 の降順に順位付けします。

```python
comparison = model.compare(
    model_names=[
        "線形回帰",
        "Ridge",
        "ランダムフォレスト回帰",
        "LightGBM",
    ],
    method="kfold",
    n_splits=5,
)

print(comparison.ranking)
print(comparison.best_model_name)

best_model = comparison.best_model
best_pred = best_model.predict(df[["x1", "x2"]])
```

`metric` を指定すると順位付け指標を変更できます。

```python
comparison = model.compare(
    model_names=["線形回帰", "Ridge", "ランダムフォレスト回帰"],
    metric="R2",
)
```

一部の候補だけ個別パラメータを指定できます。

```python
comparison = model.compare(
    model_names=["Ridge", "ランダムフォレスト回帰"],
    model_params={
        "Ridge": {"alpha": 0.1},
        "ランダムフォレスト回帰": {
            "n_estimators": 300,
            "max_depth": 8,
        },
    },
)
```

失敗した候補は、比較を継続した上で確認できます。

```python
print(comparison.failures)
```

複数目的モデルでは、共通候補または目的変数ごとの候補を指定します。

```python
comparison = multi_model.compare(
    model_names={
        "strength": ["Ridge", "ランダムフォレスト回帰"],
        "cost": ["線形回帰", "LightGBM"],
    },
    metric={
        "strength": "R2",
        "cost": "RMSE",
    },
)

print(comparison.ranking)
print(comparison.best_model_names)
```

## 学習済みモデルから逆解析

単一目的回帰では、`"min"`、`"max"`、または到達させたい数値を目的として指定します。

```python
candidates, study = model.inverse_analysis(
    "max",
    bounds_min=[0.0, 0.0],
    bounds_max=[1.0, 1.0],
    trials=300,
    n_candidate=10,
)

print(candidates)
```

指定値へ近づける場合:

```python
candidates, study = model.inverse_analysis(
    15.0,
    trials=300,
)
```

分類では、確率を高めたい学習済みクラスラベルを指定します。

```python
candidates, study = classification_model.inverse_analysis(
    "OK",
    trials=300,
)
```

複数目的では、目的変数名と方向または目標値を辞書で指定します。

```python
candidates, study = multi_model.inverse_analysis(
    {
        "strength": "max",
        "cost": "min",
    },
    sampler_type="NSGAII",
    trials=500,
    n_candidate=20,
)
```

組成比の合計制約や固定値も従来の逆解析引数をそのまま利用できます。

```python
candidates, study = multi_model.inverse_analysis(
    {
        "strength": "max",
        "cost": "min",
    },
    constraint_cols=["component_a", "component_b", "component_c"],
    constraint_value=1.0,
    fix_values=[None, None, 0.2],
    trials=500,
)
```

直近の結果はモデルにも保持されます。

```python
model.comparison_result
model.inverse_candidates
model.inverse_study
```

トップレベルの `malchan` パッケージは、重い依存関係を import 時に即座に読み込まないように、主要 API を遅延 import します。

```python
import malchan

print(malchan.__version__)
```

FastAPI / Web アプリケーション層だけを追加して使う場合:

```bash
pip install -e ".[web]"
```

```python
from malchan.app import create_app

app = create_app()
```

---

## 開発

テスト:

```bash
pytest
```

format / lint:

```bash
black src tests
ruff check src tests
```

---

## 現在の整理方針

今回の初期整理では、以下を優先します。

1. `pyproject.toml` にパッケージメタデータ、optional dependencies、pytest / ruff / black / mypy 設定を追加する。
2. `import malchan` が軽く通るように、トップレベル import を遅延化する。
3. 旧パッケージ名 `machine_learning...` を参照している import を `malchan...` または相対 import に置き換える。
4. まずは小さな import テストを追加し、以降の整理を小さな PR に分けて進める。

## ライセンス

未定です。公開前にライセンス方針を決めてください。
