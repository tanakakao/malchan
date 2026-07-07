# malchan

`malchan` は、材料・化学データを想定した機械学習パイプラインを試作するための Python パッケージです。回帰・分類モデルの学習、特徴量の前処理、説明可能性の可視化、Optuna を使った逆解析、Excel 形式での結果出力をまとめて扱うことを目指しています。

## 主な機能

- scikit-learn 互換の機械学習パイプライン作成
- 回帰・分類モデルの学習、予測、交差検証
- Optuna によるハイパーパラメータチューニング
- SHAP、Permutation Feature Importance、Partial Dependence などの可視化補助
- SMILES や組成式を使った特徴量生成の補助
- Optuna を使った逆解析・候補探索
- 学習結果や図表の Excel 出力

## インストール

現時点ではローカル開発用の最小構成です。

```bash
git clone <repository-url>
cd malchan
pip install -e .
```

必要な依存関係は、利用する機能に応じて追加してください。特に、化学・材料特徴量を使う場合は RDKit、matminer、xenonpy などの環境構築が必要になることがあります。

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

## パッケージ構成

```text
src/malchan/
├── export/              # Excel 出力関連
├── inverse_analysis/    # Optuna による逆解析
├── models/              # モデル、学習、前処理、説明可能性
└── visualization/       # Plotly ベースの可視化
```

## 開発メモ

- 関数を追加・変更する場合は Google スタイルの docstring を記述してください。
- README は試作用の初期版です。API が固まり次第、インストール方法・依存関係・サンプルを更新してください。
- 現在のコードには旧パッケージ名を参照している import が含まれる可能性があります。利用前にパッケージ名の整合性を確認してください。

## ライセンス

未定です。公開前にライセンス方針を決めてください。
