# malchan

`malchan` は、材料・化学データを想定した機械学習ワークフローを試作するための Python パッケージです。回帰・分類モデルの学習、モデル比較、特徴量の前処理、説明可能性の可視化、Optuna を使った逆解析、Excel 形式での結果出力をまとめて扱うことを目指しています。

このプロジェクトは開発初期段階です。現時点では後方互換性よりも、パッケージとして import できること、公開 API の入口を整理すること、旧ディレクトリ構成由来の import を解消することを優先しています。

---

## 主な機能

- scikit-learn 互換の機械学習パイプライン作成
- 回帰・分類モデルの学習、予測、交差検証
- Optuna によるハイパーパラメータチューニング
- SHAP、Permutation Feature Importance、Partial Dependence などの可視化補助
- SMILES や組成式を使った特徴量生成の補助
- Optuna を使った逆解析・候補探索
- 学習結果や図表の Excel 出力

---

## パッケージ構成

```text
src/malchan/
├── models/              # モデル、学習、前処理、説明可能性
├── visualization/       # Plotly ベースの可視化
├── inverse_analysis/    # Optuna による逆解析
└── export/              # Excel 出力関連
```

主な入口は以下です。

| Module | Purpose |
|---|---|
| `malchan.models` | モデルパイプライン、モデル比較、前処理、説明可能性の処理 |
| `malchan.visualization` | モデル結果・逆解析結果の可視化 |
| `malchan.inverse_analysis` | Optuna ベースの逆解析 |
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
pip install -e ".[dev,models,materials,visualization,inverse,export]"
```

すべての optional dependencies を入れる場合:

```bash
pip install -e ".[all]"
```

---

## 使い方の例

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

トップレベルの `malchan` パッケージは、重い依存関係を import 時に即座に読み込まないように、主要 API を遅延 import します。

```python
import malchan

print(malchan.__version__)
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
