# malchan.visualization の使い方

`malchan.visualization` は、学習済みの `MLModelPipeline` とその子モデルを使って、モデル評価・特徴量重要度・PD/ICE・SHAP・逆解析結果を Plotly の `go.Figure` として可視化するためのモジュールです。

## 基本方針

可視化関数は、次のどちらかの入力形式で使えます。

1. **推奨: `model` と `target` / `object_col` を渡す形式**
   - `model`: 学習済みの `MLModelPipeline`
   - `target`: `model.models` のキーになる目的変数名
   - `object_col`: `target` と同じ意味の別名
   - PD/SHAP に必要な `X_PD`, `xticks`, `X`, `y`, `shap_values` などを関数内で取得します。

2. **事前計算済みデータを渡す形式**
   - `get_pd_and_ice()` や `get_pd_2d()` の戻り値を自分で受け取り、可視化関数へ渡します。
   - 既存の解析フローや、中間データを加工してから描画したい場合に使います。

基本的には 1 の `model` ベースの形式を使うと、Notebook 上のコードを短くできます。

```python
from malchan.visualization import show_pd_2d

fig = show_pd_2d(
    model=model,
    object_col="y_cat_str",
    target_cols=["raw material 1", "raw material 3"],
)
fig.show()
```

これは、従来の次のような書き方を 1 関数呼び出しにまとめたものです。

```python
X_PD, xticks1, xticks2 = model.models["y_cat_str"].get_pd_2d(
    target_cols=["raw material 1", "raw material 3"],
)

fig = show_pd_2d(
    X_PD,
    target_cols=["raw material 1", "raw material 3"],
    xticks1=xticks1,
    xticks2=xticks2,
    X=model.models["y_cat_str"].X,
    y=model.models["y_cat_str"].y,
    col_idx=-1,
)
fig.show()
```

## `target` と `object_col` の使い分け

`target` と `object_col` はどちらも `model.models` のキーを指定するための引数です。例えば `model.models["y_cat_str"]` を可視化したい場合は、次のどちらでも同じです。

```python
show_pd_2d(model=model, target="y_cat_str", target_cols=["x1", "x2"])
show_pd_2d(model=model, object_col="y_cat_str", target_cols=["x1", "x2"])
```

プロジェクト内で目的変数名を `object_col` と呼んでいる場合は `object_col`、一般的なモデル出力名として扱う場合は `target` を使ってください。両方を同時に指定した場合は `target` が優先されます。

## 分類モデルの `col_idx`

分類モデルでは、PD/SHAP がクラス方向の次元を持つ場合があります。その場合は `col_idx` で描画対象クラスのインデックスを指定します。

```python
fig = show_pd_and_ice(
    model=model,
    object_col="y_cat_str",
    target_col="raw material 1",
    col_idx=1,
)
```

`col_idx=-1` は最後のクラスを使います。対象クラス名からインデックスを取りたい場合は、子モデルの `item2idx` を使えます。

```python
col_idx = model.models["y_cat_str"].item2idx["target_class_name"]
fig = show_pd_2d(
    model=model,
    object_col="y_cat_str",
    target_cols=["raw material 1", "raw material 3"],
    col_idx=col_idx,
)
```

## モデル評価: `yy_plot_ml`

`yy_plot_ml` は、回帰では実測値と予測値の散布図、分類では混同行列を表示します。

```python
from malchan.visualization import yy_plot_ml

fig = yy_plot_ml(
    model=model,
    object_col="yield",
)
fig.show()
```

Cross Validation の予測値を使う場合は `cv=True` を指定します。

```python
fig = yy_plot_ml(
    model=model,
    object_col="yield",
    cv=True,
    train_test="test",
)
fig.show()
```

回帰で残差を見たい場合は `residual=True` を指定します。

```python
fig = yy_plot_ml(
    model=model,
    object_col="yield",
    residual=True,
)
fig.show()
```

## 特徴量重要度: `show_importances`

`show_importances` は特徴量重要度を横棒グラフで表示します。`model` ベースで呼ぶ場合は、子モデルから重要度を取得します。

```python
from malchan.visualization import show_importances

fig = show_importances(
    model=model,
    object_col="yield",
    n_bar=20,
)
fig.show()
```

`importance_type` には、子モデルが持つ重要度メソッドの接頭辞を指定します。例えば `importance_type="model"` は `model_importance()` を呼びます。

```python
fig = show_importances(
    model=model,
    object_col="yield",
    importance_type="model",
)
```

事前に計算した重要度を渡すこともできます。

```python
fig = show_importances(
    feature_names=["x1", "x2", "x3"],
    feature_importances=[0.4, 0.1, 0.3],
    n_bar=3,
)
fig.show()
```

## 1次元 PD/ICE: `show_pd_and_ice`

`show_pd_and_ice` は、1つの特徴量に対する Partial Dependence と ICE を表示します。

```python
from malchan.visualization import show_pd_and_ice

fig = show_pd_and_ice(
    model=model,
    object_col="yield",
    target_col="temperature",
)
fig.show()
```

ICE 線を非表示にしたい場合は `ice=False` を指定します。

```python
fig = show_pd_and_ice(
    model=model,
    object_col="yield",
    target_col="temperature",
    ice=False,
)
fig.show()
```

事前計算済みデータを使う場合は、次のように `X_PD`, `xticks`, `X`, `y` を渡します。

```python
X_PD, xticks = model.models["yield"].get_pd_and_ice("temperature")

fig = show_pd_and_ice(
    X_PD=X_PD,
    target_col="temperature",
    xticks=xticks,
    X=model.models["yield"].X,
    y=model.models["yield"].y,
)
fig.show()
```

## 2次元 PD: `show_pd_2d`

`show_pd_2d` は、2つの特徴量に対する Partial Dependence を等高線として表示します。

```python
from malchan.visualization import show_pd_2d

fig = show_pd_2d(
    model=model,
    object_col="y_cat_str",
    target_cols=["raw material 1", "raw material 3"],
)
fig.show()
```

事前計算済みデータを使う場合は、次のように呼びます。

```python
X_PD, xticks1, xticks2 = model.models["y_cat_str"].get_pd_2d(
    target_cols=["raw material 1", "raw material 3"],
)

fig = show_pd_2d(
    X_PD=X_PD,
    target_cols=["raw material 1", "raw material 3"],
    xticks1=xticks1,
    xticks2=xticks2,
    X=model.models["y_cat_str"].X,
    y=model.models["y_cat_str"].y,
)
fig.show()
```

## SHAP 散布図: `show_shap_scatter`

`show_shap_scatter` は、1つの特徴量とその SHAP 値の関係を散布図で表示します。

```python
from malchan.visualization import show_shap_scatter

fig = show_shap_scatter(
    model=model,
    object_col="yield",
    target_col="temperature",
)
fig.show()
```

別の特徴量で色分けしたい場合は `interactive_col` を指定します。

```python
fig = show_shap_scatter(
    model=model,
    object_col="yield",
    target_col="temperature",
    interactive_col="catalyst",
)
fig.show()
```

事前計算済みデータを使う場合は、子モデルの `get_shap_scatter_data()` の結果を渡します。

```python
X_shappd = model.models["yield"].get_shap_scatter_data("temperature")

fig = show_shap_scatter(
    X_shappd=X_shappd,
    rawX=model.models["yield"].X,
    shap_values=model.models["yield"].shap_values,
    target_col="temperature",
    unique_dict=model.models["yield"]._shared_attr("unique_cols"),
)
fig.show()
```

## SHAP ビーズウォーム: `show_shap_beeswarm`

`show_shap_beeswarm` は、SHAP 値が大きい上位特徴量をビーズウォーム形式で表示します。

```python
from malchan.visualization import show_shap_beeswarm

fig = show_shap_beeswarm(
    model=model,
    object_col="yield",
    n_shap_top=10,
)
fig.show()
```

事前計算済みの `X` と `shap_values` を渡すこともできます。

```python
fig = show_shap_beeswarm(
    X=model.models["yield"].X,
    shap_values=model.models["yield"].shap_values,
    n_shap_top=10,
    cat_cols=model.models["yield"]._shared_attr("cat_cols"),
)
fig.show()
```

## 逆解析結果の可視化

逆解析向けの関数は、候補点や Pareto プロットを重ねて表示するための補助関数です。

### `show_ia_result_with_pd`

1次元 PD 上に逆解析の候補点を重ねます。

```python
from malchan.visualization import show_ia_result_with_pd

fig = show_ia_result_with_pd(
    df_trials=df_trials,
    model=model,
    target_col="yield",
    feature_col="temperature",
    target_item=None,
)
fig.show()
```

### `show_ia_result_with_pd_2d`

2次元 PD 上に逆解析の候補点を重ねます。

```python
from malchan.visualization import show_ia_result_with_pd_2d

fig = show_ia_result_with_pd_2d(
    df_trials=df_trials,
    model=model,
    target_col="yield",
    feature_cols=["temperature", "pressure"],
    target_item=None,
)
fig.show()
```

### `show_ia_importance`

Optuna の study から逆解析パラメータの重要度を表示します。

```python
from malchan.visualization import show_ia_importance

fig = show_ia_importance(
    model=model,
    study=study,
    target="yield",
)
fig.show()
```

### `show_ia_result_pareto`

2つの目的変数について、学習データと逆解析候補点を Pareto プロットとして表示します。

```python
from malchan.visualization import show_ia_result_pareto

fig = show_ia_result_pareto(
    model=model,
    df_trials=df_trials,
    target1="yield",
    target2="cost",
)
fig.show()
```

分類目的変数で特定クラスの確率を使いたい場合は `target_item1` / `target_item2` を指定します。

```python
fig = show_ia_result_pareto(
    model=model,
    df_trials=df_trials,
    target1="class_a",
    target2="cost",
    target_item1="OK",
)
fig.show()
```

## よくあるエラー

### `model` を渡したのに `target` / `object_col` が無い

`model` ベースで呼ぶ場合は、どの子モデルを使うか指定する必要があります。

```python
# NG
show_pd_2d(model=model, target_cols=["x1", "x2"])

# OK
show_pd_2d(model=model, object_col="yield", target_cols=["x1", "x2"])
```

### `target_cols` は2つ指定する

`show_pd_2d` は2次元PD用の関数なので、x軸とy軸の2列を指定してください。

```python
show_pd_2d(
    model=model,
    object_col="yield",
    target_cols=["temperature", "pressure"],
)
```

### SHAP 値が計算されていない

`show_shap_scatter` と `show_shap_beeswarm` は、子モデルの `shap_values` が必要です。SHAP を計算していないモデルでは `ValueError` になります。モデル学習後に SHAP 計算を含む処理を実行してから呼び出してください。

## Notebook での表示

各関数は Plotly の `go.Figure` を返します。Notebook では `fig.show()`、Streamlit では `st.plotly_chart(fig)` などで表示できます。

```python
fig = show_pd_2d(
    model=model,
    object_col="yield",
    target_cols=["temperature", "pressure"],
)
fig.show()
```
