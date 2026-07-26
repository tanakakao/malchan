# malchan FastAPI + React application

`malchan.app`は、モデル学習・予測・比較・交差検証・ベストモデルチューニング・XAI・逆解析を提供するFastAPIと、Reactワークベンチをまとめたアプリケーション層です。

## Python / JupyterからDataFrameを使ってFastAPIを利用

FastAPIを外部サーバーとして起動せず、PythonコードやJupyter Notebookのプロセス内で直接利用できます。Notebook上では`pandas.DataFrame`を主なデータ形式として扱い、APIへ送信する直前に行指向のJSONレコードへ変換します。

```bash
pip install -e ".[api,notebook,models,visualization,inverse]"
```

### TestClientを作成

`create_app()`はアプリケーションファクトリです。NotebookではReact配信とCORSを無効にし、APIだけをプロセス内で利用できます。

```python
from fastapi.testclient import TestClient

from malchan.app import AppSettings, create_app


settings = AppSettings(
    api_prefix="/api",
    cors_origins=(),
    serve_frontend=False,
)
app = create_app(
    settings=settings,
    title="malchan Notebook API",
)
client = TestClient(app)

health_response = client.get("/api/health")
health_response.raise_for_status()
print(health_response.json())
```

`TestClient`は実際のポートを使用しません。同じ`app`と`client`を使い続ける限り、学習済みモデル、比較結果、XAIキャッシュは同じ`InMemoryModelService`に保持されます。

### DataFrameをAPIレコードへ変換

FastAPIのrequest bodyはJSONです。`dataframe_to_records()`は、NumPy・pandasのスカラー値、`NaN`・`pd.NA`・`NaT`、日時列をJSON互換値へ正規化します。

```python
import pandas as pd

from malchan.app import dataframe_to_records


df = pd.read_csv("resin.csv")
records = dataframe_to_records(df)
```

DataFrameの列名は重複のない文字列にしてください。単純なDataFrameでは`df.to_dict(orient="records")`も使えますが、欠損値やpandas固有型を含む実データでは`dataframe_to_records()`を推奨します。

## Notebookでモデルを学習して予測

```python
input_cols = [
    "raw material 1",
    "raw material 2",
    "raw material 3",
    "temperature",
    "time",
]
target_col = "property"

train_response = client.post(
    "/api/models",
    json={
        "data": records,
        "target_col": target_col,
        "task": "regression",
        "num_cols": input_cols,
        "cat_cols": [],
        "model_names": ["ランダムフォレスト回帰"],
        "compute_xai": False,
    },
)
train_response.raise_for_status()

model_info = train_response.json()
model_id = model_info["model_id"]
print(model_info)
```

予測データもDataFrameから送信します。

```python
predict_df = df[input_cols].iloc[:5].copy()

predict_response = client.post(
    f"/api/models/{model_id}/predict",
    json={"data": dataframe_to_records(predict_df)},
)
predict_response.raise_for_status()

prediction_df = pd.DataFrame(predict_response.json()["predictions"])
display(prediction_df)
```

## Notebookでモデル比較と交差検証を利用

`POST /api/models/{model_id}/compare`は、登録モデルが保持する学習データと前処理設定を再利用し、候補モデルを同じ交差検証条件で比較します。

### K-fold CVで候補モデルを比較

```python
compare_response = client.post(
    f"/api/models/{model_id}/compare",
    json={
        "model_names": [
            "線形回帰",
            "Ridge",
            "ランダムフォレスト回帰",
            "LightGBM",
        ],
        "method": "kfold",
        "n_splits": 5,
        "metric": "RMSE",
        "tuning": False,
        "tune_best": False,
        "continue_on_error": True,
        "activate_best": False,
    },
)
compare_response.raise_for_status()
comparison = compare_response.json()
```

目的変数ごとのランキングをDataFrameで確認します。

```python
target_result = comparison["targets"][target_col]
ranking_df = pd.DataFrame(target_result["ranking"])

display(ranking_df)
print("metric:", target_result["metric"])
print("higher_is_better:", target_result["higher_is_better"])
print("best model:", target_result["best_model_name"])
print("failures:", target_result["failures"])
```

`higher_is_better=False`の指標では小さい値が上位、`True`の指標では大きい値が上位です。`continue_on_error=True`では、一部の候補が失敗しても比較を続行し、理由を`failures`へ記録します。

### CV方式

| `method` | 内容 | 主な用途 |
|---|---|---|
| `kfold` | データを`n_splits`個に分割して評価 | 通常のモデル選定 |
| `loo` | 1サンプルずつ検証データにするLeave-One-Out | 小規模データでの詳細評価 |

LOOの例です。

```python
loo_response = client.post(
    f"/api/models/{model_id}/compare",
    json={
        "model_names": ["Ridge", "ランダムフォレスト回帰"],
        "method": "loo",
        "metric": "RMSE",
        "continue_on_error": True,
    },
)
loo_response.raise_for_status()
```

LOOはデータ行数と同じ回数だけ学習するため、データ数やモデルによっては計算量が大きくなります。

### 指標を変更

回帰では`RMSE`や`R2`、分類では`F1`など、モデル比較処理が対応する指標を指定できます。

```python
compare_response = client.post(
    f"/api/models/{model_id}/compare",
    json={
        "model_names": ["Ridge", "ランダムフォレスト回帰"],
        "method": "kfold",
        "n_splits": 5,
        "metric": "R2",
    },
)
compare_response.raise_for_status()
```

複数目的モデルでは、候補と指標を目的変数ごとの辞書で指定できます。

```python
multi_compare_response = client.post(
    f"/api/models/{model_id}/compare",
    json={
        "model_names": {
            "strength": ["Ridge", "ランダムフォレスト回帰"],
            "cost": ["線形回帰", "LightGBM"],
        },
        "metric": {
            "strength": "R2",
            "cost": "RMSE",
        },
        "method": "kfold",
        "n_splits": 5,
    },
)
```

### チューニング方法の違い

| 設定 | 動作 | 計算量 |
|---|---|---|
| `tuning=False`, `tune_best=False` | チューニングせず公平に比較 | 小 |
| `tune_best=True` | 比較後、最良候補だけOptunaでチューニング | 中 |
| `tuning=True` | 全候補をチューニングしてから比較 | 大 |

`tuning=True`と`tune_best=True`は同時に指定できません。

比較と最良候補のチューニングを1回で行う例です。

```python
compare_response = client.post(
    f"/api/models/{model_id}/compare",
    json={
        "model_names": ["Ridge", "ランダムフォレスト回帰", "LightGBM"],
        "method": "kfold",
        "n_splits": 5,
        "metric": "RMSE",
        "tune_best": True,
        "tuning_trials": 50,
        "tuning_verbose": 0,
        "activate_best": False,
    },
)
compare_response.raise_for_status()
```

### 比較後に最良モデルだけチューニング

先にランキングを確認し、その後でチューニングできます。

```python
tune_response = client.post(
    f"/api/models/{model_id}/comparison/tune-best",
    json={
        "targets": [],
        "n_trials": 50,
        "verbose": 0,
        "evaluate": True,
        "activate_best": False,
    },
)
tune_response.raise_for_status()
tuned_comparison = tune_response.json()
```

`targets=[]`は全目的変数を意味します。複数目的で一部だけチューニングする場合は`["strength"]`のように指定します。`evaluate=True`では、チューニング後の最良モデルを再度CV評価します。

### チューニング後のCVスコア

`best_cv_scores`は、チューニング後に`evaluate=True`で再評価した場合に返されます。分割ごとのレコードをDataFrameへ変換できます。

```python
target_result = tuned_comparison["targets"][target_col]
print("best model:", target_result["best_model_name"])
print("best params:", target_result["best_params"])
print("best is tuned:", target_result["best_is_tuned"])

best_cv_scores = target_result["best_cv_scores"] or {}
for split_name, score_records in best_cv_scores.items():
    print(split_name)
    display(pd.DataFrame(score_records))
```

このHTTP APIが返すのはCVの評価スコアです。サンプルごとのCV予測値は現在のレスポンスには含まれません。Pythonでモデルオブジェクトを直接扱う場合は、`yy_plot_ml(model, target=..., cv=True)`でモデル内部のCV予測を可視化できます。

### 最良モデルを後続処理へ反映

`activate_best=True`を指定すると、同じ`model_id`に登録されているモデルが最良モデルへ置き換わります。以後の予測・XAI・逆解析は有効化されたモデルを使用します。

```python
activate_response = client.post(
    f"/api/models/{model_id}/comparison/tune-best",
    json={
        "targets": [],
        "n_trials": 50,
        "evaluate": True,
        "activate_best": True,
    },
)
activate_response.raise_for_status()
```

学習時にXAIを要求していたモデルでは、最良モデルを有効化した後にXAIキャッシュも更新されます。

### 最新の比較結果を再取得

```python
comparison_response = client.get(
    f"/api/models/{model_id}/comparison"
)
comparison_response.raise_for_status()
comparison = comparison_response.json()
```

比較を一度も実行していない場合はHTTP `409`です。

## Notebookで逆解析を利用

逆解析では、登録済みモデルを固定した予測器として使い、Optunaで入力条件を探索します。比較後に`activate_best=True`で最良モデルを有効化している場合は、そのモデルが探索に使用されます。

```text
POST /api/models/{model_id}/inverse-analysis
```

### 目的の指定方法

目的変数ごとに、`direction`または`target_value`のどちらか一方を指定します。

| 指定 | 内容 | 例 |
|---|---|---|
| `direction="max"` | 予測値を最大化 | 強度を高くする |
| `direction="min"` | 予測値を最小化 | コストを低くする |
| `target_value=<value>` | 指定値との差を小さくする | 物性を50へ近づける |
| `target_value=<class>` | 指定クラスの確率を高くする | 品質を`OK`へ近づける |

`direction`と`target_value`を同時に指定したり、両方を省略したりするとHTTP `422`です。分類目的では`direction`ではなく、学習時に存在したクラスラベルを`target_value`へ指定します。

### 単一目的を最大化

```python
inverse_response = client.post(
    f"/api/models/{model_id}/inverse-analysis",
    json={
        "objectives": [
            {
                "target": target_col,
                "direction": "max",
            }
        ],
        "sampler_type": "TPE",
        "trials": 500,
        "n_candidates": 20,
    },
)
inverse_response.raise_for_status()

inverse_result = inverse_response.json()
candidates_df = pd.DataFrame(inverse_result["candidates"])
display(candidates_df)
```

`bounds`や`categories`を省略した場合、数値特徴量は学習データの最小値から最大値、カテゴリ特徴量は学習データで観測されたユニーク値を使います。

### 指定値へ近づける

```python
inverse_response = client.post(
    f"/api/models/{model_id}/inverse-analysis",
    json={
        "objectives": [
            {
                "target": target_col,
                "target_value": 50.0,
            }
        ],
        "sampler_type": "TPE",
        "trials": 500,
        "n_candidates": 20,
    },
)
inverse_response.raise_for_status()
```

### 数値探索範囲と刻み幅

`bounds`は数値特徴量ごとに設定します。

```python
bounds = {
    "raw material 1": {
        "min": 0.0,
        "max": 100.0,
        "dtype": "float",
        "step": 0.1,
    },
    "raw material 2": {
        "min": 0.0,
        "max": 100.0,
        "dtype": "float",
        "step": 0.1,
    },
    "raw material 3": {
        "min": 0.0,
        "max": 100.0,
        "dtype": "float",
        "step": 0.1,
    },
    "temperature": {
        "min": 120.0,
        "max": 200.0,
        "dtype": "float",
        "step": 1.0,
    },
    "time": {
        "min": 10,
        "max": 120,
        "dtype": "int",
        "step": 5,
    },
}
```

`dtype="int"`の場合、`min`、`max`、`step`は整数である必要があります。`dtype`を省略すると学習データの型から推定され、整数列の`step`を省略した場合は1になります。

### カテゴリ候補を限定

カテゴリ、SMILES、組成式列の探索候補は`categories`で限定できます。

```python
categories = {
    "catalyst": ["A", "B", "C"],
    "solvent": ["water", "ethanol"],
}
```

指定しないカテゴリ列では学習データに存在する値を使用します。空の候補リストは指定できません。

### 一部の特徴量を固定

```python
fixed_values = {
    "temperature": 160.0,
    "catalyst": "B",
}
```

固定した特徴量はOptunaの探索対象から外れます。固定値はモデルが受け付ける型と値に合わせてください。

### 合計制約

配合比など、複数の数値特徴量の合計を固定できます。

```python
sum_constraint = {
    "columns": [
        "raw material 1",
        "raw material 2",
        "raw material 3",
    ],
    "value": 100.0,
}
```

合計制約に指定できるのは数値特徴量だけです。同じ列を重複して指定できません。各列の上下限と固定値によって合計値を実現できない設定もあるため、返された候補の合計を確認してください。

### 範囲・カテゴリ・固定値・合計制約をまとめて指定

```python
inverse_response = client.post(
    f"/api/models/{model_id}/inverse-analysis",
    json={
        "objectives": [
            {
                "target": target_col,
                "direction": "max",
            }
        ],
        "sampler_type": "TPE",
        "bounds": bounds,
        "categories": categories,
        "fixed_values": fixed_values,
        "sum_constraint": sum_constraint,
        "trials": 1000,
        "n_candidates": 30,
    },
)
inverse_response.raise_for_status()

inverse_result = inverse_response.json()
candidates_df = pd.DataFrame(inverse_result["candidates"])
display(candidates_df)
```

リクエストで指定していない数値列・カテゴリ列も探索対象です。探索させたくない列は`fixed_values`へ指定してください。

### 分類モデルで指定クラスを探索

```python
classification_response = client.post(
    f"/api/models/{model_id}/inverse-analysis",
    json={
        "objectives": [
            {
                "target": "quality",
                "target_value": "OK",
            }
        ],
        "sampler_type": "TPE",
        "trials": 500,
        "n_candidates": 20,
    },
)
classification_response.raise_for_status()

classification_candidates_df = pd.DataFrame(
    classification_response.json()["candidates"]
)
display(classification_candidates_df)
```

存在しないクラスラベルを指定するとHTTP `422`です。分類の候補DataFrameでは、クラス確率が`pred_<target>_<class>`形式の列として含まれる場合があります。実際の列名は`candidates_df.columns`で確認してください。

### 多目的逆解析

複数目的では、それぞれの目的を同時に指定します。

```python
multi_inverse_response = client.post(
    f"/api/models/{model_id}/inverse-analysis",
    json={
        "objectives": [
            {
                "target": "strength",
                "direction": "max",
            },
            {
                "target": "cost",
                "direction": "min",
            },
        ],
        "sampler_type": "NSGAII",
        "bounds": bounds,
        "trials": 2000,
        "n_candidates": 50,
    },
)
multi_inverse_response.raise_for_status()

multi_inverse_result = multi_inverse_response.json()
pareto_candidates_df = pd.DataFrame(multi_inverse_result["candidates"])
display(pareto_candidates_df)
```

候補は、各目的を正規化した合計スコアによって順位付けされます。レスポンスの`pareto_size`はOptuna Study内のPareto最良試行数です。

### サンプラー

| `sampler_type` | 主な用途の目安 |
|---|---|
| `TPE` | 単一目的の標準的な探索 |
| `MOTPE` | TPE系の多目的探索 |
| `CmaEs` | 主に連続値中心の単一目的探索 |
| `GP` | ガウス過程ベースの探索 |
| `QMS` | 準モンテカルロ探索 |
| `NSGAII` | 多目的Pareto探索 |
| `NSGAIII` | 目的数が多い多目的探索 |

カテゴリ変数、整数変数、固定値、制約との組み合わせによって適したサンプラーは異なります。まず`TPE`、多目的では`NSGAII`を基準に比較してください。

### レスポンスを確認

```python
inverse_result = inverse_response.json()

print("model_id:", inverse_result["model_id"])
print("objectives:", inverse_result["objectives"])
print("requested trials:", inverse_result["n_trials"])
print("completed trials:", inverse_result["n_completed_trials"])
print("pareto size:", inverse_result["pareto_size"])

candidates_df = pd.DataFrame(inverse_result["candidates"])
display(candidates_df)
```

候補DataFrameには、入力特徴量と`pred_<target>`形式の予測列が含まれます。分類確率を返す場合は`pred_<target>_<class>`形式になることがあります。

`n_candidates`は`trials`以下である必要があります。`n_completed_trials`が少ない場合は、探索範囲、カテゴリ候補、固定値、合計制約、モデル予測時のエラーを確認してください。

### 単一目的候補を可視化

FastAPIレスポンスはDataFrameとしてPlotlyへ渡せます。

```python
import plotly.express as px


plot_df = candidates_df.copy()
plot_df["candidate"] = range(1, len(plot_df) + 1)
prediction_col = f"pred_{target_col}"

fig = px.scatter(
    plot_df,
    x="candidate",
    y=prediction_col,
    hover_data=[column for column in input_cols if column in plot_df.columns],
    title=f"Inverse-analysis candidates: {target_col}",
)
fig.show()
```

指定値へ近づける探索では、目標値を水平線として追加できます。

```python
fig.add_hline(
    y=50.0,
    line_dash="dash",
    annotation_text="target value",
)
fig.show()
```

### 多目的候補をPareto散布図で確認

```python
fig = px.scatter(
    pareto_candidates_df,
    x="pred_strength",
    y="pred_cost",
    hover_data=[
        column
        for column in input_cols
        if column in pareto_candidates_df.columns
    ],
    title="Inverse-analysis candidates: strength vs cost",
)
fig.show()
```

APIは上位候補と集計情報を返しますが、Optuna Studyオブジェクト自体は返しません。そのため、`show_ia_importance()`のようにStudyを必要とする既存可視化関数は、Pythonでモデルと`inverse_analysis()`を直接扱う場合に使用します。

### Pythonモデルを直接使う場合の可視化

```python
from malchan.inverse_analysis import inverse_analysis
from malchan.visualization import (
    show_ia_importance,
    show_ia_result_pareto,
)


candidates_df, study = inverse_analysis(
    model=model,
    sampler_type="NSGAII",
    obj_directions=["max", "min"],
    target_cols=["strength", "cost"],
    trials=2000,
    n_candidate=50,
)

show_ia_importance(
    model=model,
    study=study,
    target="strength",
).show()

show_ia_result_pareto(
    model=model,
    df_trials=candidates_df,
    target1="strength",
    target2="cost",
).show()
```

`show_ia_result_with_pd()`と`show_ia_result_with_pd_2d()`も、学習済みモデルを直接保持するPythonワークフロー向けです。

### 逆解析で主に発生するHTTPエラー

| Status | 主な原因 |
|---|---|
| `404` | `model_id`が存在しない |
| `422` | 目的変数、クラス、特徴量名、範囲、固定値、カテゴリ候補、制約が不正 |

HTTP `422`の内容は次のように確認できます。

```python
if inverse_response.status_code == 422:
    print(inverse_response.json()["detail"])
else:
    inverse_response.raise_for_status()
```

## NotebookでXAIを利用

XAIを利用する場合は、学習リクエストで`compute_xai=True`を指定します。SHAP、重要度、PDP/ICEは学習後に一度計算され、以後のGET APIは保存済みキャッシュを返します。

```python
from malchan.visualization import (
    show_xai_importance,
    show_xai_pd_and_ice,
    show_xai_shap_beeswarm,
    show_xai_shap_scatter,
)
```

| 関数 | 対応するエンドポイント | 用途 |
|---|---|---|
| `show_xai_importance()` | `GET .../importance` | 特徴量重要度 |
| `show_xai_shap_beeswarm()` | `GET .../shap-values` | 全特徴量のSHAP分布 |
| `show_xai_shap_scatter()` | `GET .../shap?feature=...` | 特徴量別SHAP dependence |
| `show_xai_pd_and_ice()` | `GET .../pdp?feature=...` | PDP・ICE |

### XAIを有効にして学習

```python
xai_train_response = client.post(
    "/api/models",
    json={
        "data": records,
        "target_col": target_col,
        "task": "regression",
        "num_cols": input_cols,
        "cat_cols": [],
        "model_names": ["ランダムフォレスト回帰"],
        "compute_xai": True,
    },
)
xai_train_response.raise_for_status()

model_info = xai_train_response.json()
model_id = model_info["model_id"]
print("xai_status:", model_info["xai_status"])
```

XAI計算で失敗してもモデル登録自体は成功します。目的変数ごとの状態とエラーはXAIサマリーで確認します。

```python
summary_response = client.get(f"/api/models/{model_id}/xai")
summary_response.raise_for_status()
xai_summary = summary_response.json()

summary_df = pd.DataFrame(
    [
        {
            "target": target,
            "status": info["status"],
            "computed_at": info["computed_at"],
            "error": info["error"],
            "importance_methods": ", ".join(info["importance_methods"]),
            "shap_features": ", ".join(info["shap_features"]),
            "pdp_features": ", ".join(info["pdp_features"]),
        }
        for target, info in xai_summary["targets"].items()
    ]
)
display(summary_df)

target = next(
    name
    for name, info in xai_summary["targets"].items()
    if info["status"] == "ready"
)
target_info = xai_summary["targets"][target]
```

### 特徴量重要度

`method`には`model`、`pfi`、`shap`を指定できます。

```python
importance_response = client.get(
    f"/api/models/{model_id}/xai/{target}/importance",
    params={
        "method": "shap",
        "combined": True,
        "top_n": 20,
    },
)
importance_response.raise_for_status()

show_xai_importance(importance_response, n_bar=20).show()
importance_df = pd.DataFrame(importance_response.json()["items"])
display(importance_df)
```

`combined=True`では、可能な場合にone-hot encoding後の列などを元の入力列単位へ集約します。集約キャッシュがない場合は前処理後の特徴量へフォールバックし、レスポンスの`combined`が`False`になります。

### 全特徴量のSHAP値をまとめて取得

`GET .../shap-values`は、全サンプルの元特徴量値と、列が揃ったSHAP行列を一度に返します。特徴量別APIを繰り返し呼ぶ必要はありません。

```python
shap_values_response = client.get(
    f"/api/models/{model_id}/xai/{target}/shap-values"
)
shap_values_response.raise_for_status()
shap_values_payload = shap_values_response.json()

print("features:", shap_values_payload["features"])
print("outputs:", shap_values_payload["output_names"])
```

レスポンスの主なフィールドです。

| Field | 内容 |
|---|---|
| `features` | SHAP行列の列順 |
| `cat_cols` | カテゴリ特徴量 |
| `records` | 全特徴量の元データ行 |
| `output_names` | `shap`または`shap_<class>` |
| `shap_values` | output名ごとの`n_samples × n_features`行列 |

DataFrameとして取り出せます。

```python
shap_X_df = pd.DataFrame(shap_values_payload["records"])[
    shap_values_payload["features"]
]

shap_matrix_dfs = {
    output_name: pd.DataFrame(
        matrix,
        columns=shap_values_payload["features"],
    )
    for output_name, matrix in shap_values_payload["shap_values"].items()
}

display(shap_X_df.head())
for output_name, shap_matrix_df in shap_matrix_dfs.items():
    print(output_name)
    display(shap_matrix_df.head())
```

回帰では通常`output_names=["shap"]`です。分類では`shap_OK`、`shap_NG`のようなクラス別行列を返す場合があります。

### SHAP Beeswarm

```python
beeswarm_fig = show_xai_shap_beeswarm(
    shap_values_response,
    n_shap_top=15,
)
beeswarm_fig.show()
```

分類で表示するクラスを指定する場合:

```python
beeswarm_fig = show_xai_shap_beeswarm(
    shap_values_response,
    n_shap_top=15,
    target_item="OK",
)
beeswarm_fig.show()
```

`target_item`を省略した複数クラス応答では、最後のSHAP出力を表示します。解釈を明確にするため、分類ではクラス名を明示することを推奨します。

### 特徴量別SHAP散布図

全体傾向はBeeswarm、1特徴量の値とSHAP値の関係は特徴量別`/shap`を使用します。

```python
shap_feature = target_info["shap_features"][0]

shap_response = client.get(
    f"/api/models/{model_id}/xai/{target}/shap",
    params={"feature": shap_feature},
)
shap_response.raise_for_status()

show_xai_shap_scatter(shap_response).show()
```

分類で1クラスだけ表示する場合:

```python
show_xai_shap_scatter(
    shap_response,
    target_item="OK",
).show()
```

### PDPとICE

```python
pdp_feature = target_info["pdp_features"][0]

pdp_response = client.get(
    f"/api/models/{model_id}/xai/{target}/pdp",
    params={
        "feature": pdp_feature,
        "include_ice": True,
        "max_ice": 30,
    },
)
pdp_response.raise_for_status()

show_xai_pd_and_ice(
    pdp_response,
    ice=True,
    max_ice=30,
).show()
```

分類で特定クラスだけ表示する場合:

```python
show_xai_pd_and_ice(
    pdp_response,
    series_name="OK",
    ice=True,
).show()
```

PDPだけ表示する場合は`include_ice=False`で取得するか、可視化時に`ice=False`を指定します。

### XAIを再計算

```python
recompute_response = client.post(
    f"/api/models/{model_id}/xai/recompute",
    json={"targets": [target]},
)
recompute_response.raise_for_status()
```

全目的変数を再計算する場合は`targets=[]`です。

### XAI状態

- `not_requested`: 学習時に無効化され、まだ計算していない
- `computing`: 計算中
- `ready`: キャッシュが利用可能
- `partial`: 複数目的のうち一部だけ成功
- `failed`: XAI計算に失敗
- `unavailable`: モデルが必要なXAI処理を提供していない

XAI取得時の主なHTTPステータスです。

| Status | 主な原因 |
|---|---|
| `200` | キャッシュを正常に取得 |
| `404` | `model_id`が存在しない |
| `409` | XAIを未計算、計算失敗、要求したキャッシュがない |
| `422` | 目的変数、特徴量、手法、query parameterが不正 |

## インメモリ状態のライフサイクル

- モデル、比較結果、XAIキャッシュは同じ`app`内のメモリに保持されます。
- `create_app()`または`TestClient`を作り直すと、以前の状態は引き継がれません。
- 逆解析結果とOptuna Studyはサーバー側へ永続保存されません。必要な候補はNotebook側で保存してください。
- XAIのGET APIは再計算せず、保存済みキャッシュを返します。
- `POST /xai/recompute`は選択した目的変数のXAIを再構築します。
- `activate_best=True`でモデルを置き換えた場合、XAIが要求済みなら新しいモデルのXAIも更新します。
- `DELETE /api/models/{model_id}`でモデル、比較状態、XAIキャッシュを削除します。

処理後に明示的に閉じる場合:

```python
client.close()
```

## 外部サーバーとして起動

```python
# main.py
from malchan.app import AppSettings, create_app

app = create_app(
    settings=AppSettings(serve_frontend=False),
    title="malchan API",
)
```

```bash
uvicorn main:app --reload
```

またはアプリケーションファクトリを直接指定します。

```bash
MALCHAN_SERVE_FRONTEND=false \
  uvicorn "malchan.app:create_app" --factory --reload
```

## React開発

ターミナル1:

```bash
pip install -e ".[web,models,inverse,visualization,test]"
uvicorn "malchan.app:create_app" --factory --reload
```

ターミナル2:

```bash
cd frontend
npm install
npm run dev
```

Viteは`/api`をFastAPIへプロキシします。

- React開発UI: `http://127.0.0.1:5173/`
- OpenAPI: `http://127.0.0.1:8000/docs`
- API: `http://127.0.0.1:8000/api`

## FastAPIからReactを配信

```bash
cd frontend
npm install
npm run build
cd ..
uvicorn "malchan.app:create_app" --factory
```

Viteの出力先は`src/malchan/app/web/static`です。ビルド後はWeb UIを`http://127.0.0.1:8000/`から利用できます。

## 画面構成

| Step | React画面 | 主な処理 |
|---|---|---|
| 1 | Data | CSV/XLSX読込、型推定、プレビュー、欠損・列統計 |
| 2 | Explore | ヒストグラム、散布図、相関ヒートマップ |
| 3 | Prepare | 単一・複数目的、回帰/分類、説明変数選択 |
| 4 | Model | 学習、候補比較、CV、最良モデルチューニング、有効化 |
| 5 | Explain | Y-Y図、残差図、重要度、SHAP Beeswarm、SHAP散布、PDP/ICE |
| 6 | Optimize | 目的、探索範囲、カテゴリ候補、固定値、合計制約を使った逆解析 |
| 7 | Report | 分析レポート用プロンプト生成 |

## APIエンドポイント

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/health` | 稼働確認 |
| `POST` | `/api/models` | モデル学習、XAI事前計算、登録 |
| `GET` | `/api/models` | 登録モデル一覧 |
| `GET` | `/api/models/{model_id}` | モデル情報と`xai_status` |
| `POST` | `/api/models/{model_id}/predict` | 予測・分類確率 |
| `POST` | `/api/models/{model_id}/compare` | 候補モデルのCV比較と任意チューニング |
| `GET` | `/api/models/{model_id}/comparison` | 最新比較結果 |
| `POST` | `/api/models/{model_id}/comparison/tune-best` | 最良候補の追加チューニング・有効化 |
| `POST` | `/api/models/{model_id}/inverse-analysis` | Optunaによる単一・多目的逆解析 |
| `GET` | `/api/models/{model_id}/xai` | XAI状態と利用可能データ |
| `GET` | `/api/models/{model_id}/xai/{target}/importance` | モデル/PFI/SHAP重要度 |
| `GET` | `/api/models/{model_id}/xai/{target}/shap-values` | 全特徴量の元データとSHAP行列 |
| `GET` | `/api/models/{model_id}/xai/{target}/shap` | 特徴量別SHAP散布用データ |
| `GET` | `/api/models/{model_id}/xai/{target}/pdp` | 特徴量別PDPと任意のICE |
| `POST` | `/api/models/{model_id}/xai/recompute` | XAI再計算 |
| `DELETE` | `/api/models/{model_id}` | モデルと関連キャッシュ削除 |

## 現在の制約

- 学習、CV比較、チューニング、XAI、逆解析は同期処理です。
- 逆解析APIは上位候補と集計情報を返しますが、Optuna Studyや全試行履歴は返しません。
- 逆解析の探索時間は、試行数、目的数、モデル推論時間、探索変数数に応じて増えます。
- 全SHAPレスポンスは`n_samples × n_features`の行列を含むため、大規模データではレスポンスサイズが増えます。
- XAI事前計算は全特徴量のSHAP・PDP/ICEを保持するため、特徴量数・データ数に応じて学習時間とメモリ使用量が増えます。
- FastAPIの比較レスポンスはCVスコアを返しますが、サンプルごとのCV予測値はまだ返しません。
- モデル、比較状態、XAIキャッシュはプロセス内メモリに保持され、複数workerでは共有されません。
- 大規模ファイル向けのストリーミングアップロードは未実装です。
