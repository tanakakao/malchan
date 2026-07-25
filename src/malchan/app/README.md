# malchan FastAPI + React application

`malchan.app`は、モデル学習・予測・比較・ベストモデルチューニング・XAI・逆解析を提供するFastAPIと、スタンドアロンHTMLを参考にしたReactワークベンチをまとめたアプリケーション層です。

## Python / JupyterからDataFrameを使ってFastAPIを利用

FastAPIアプリを外部サーバーとして起動せず、PythonコードやJupyter Notebookのプロセス内で直接利用できます。Notebook上では`pandas.DataFrame`を主なデータ形式として扱い、APIへ送信する直前に行指向のJSONレコードへ変換します。

`TestClient`を利用するため、`api` extraにはFastAPI、Uvicorn、Pydantic、HTTPXをまとめています。

```bash
pip install -e ".[api,notebook]"
```

`create_app()`はアプリケーションファクトリです。NotebookではReact配信とCORSを無効にした設定を渡すと、APIだけをプロセス内で利用できます。

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
    version="0.1.0-notebook",
)
client = TestClient(app)

response = client.get("/api/health")
response.raise_for_status()
print(response.json())
```

`TestClient`は実際のポートを使用しないため、Notebookのイベントループや既存サーバーと競合しません。同じ`app`と`client`を使い続ける限り、学習済みモデルは`InMemoryModelService`に保持されます。

### DataFrameからモデルを学習して予測

学習データと予測データはDataFrameで準備します。FastAPIのrequest bodyはJSONであるため、`dataframe_to_records()`で`data`フィールドへ格納できる形式に変換します。

`dataframe_to_records()`は、NumPy・pandasのスカラー値、`NaN`・`pd.NA`・`NaT`、日時列をJSON互換値へ正規化します。APIの列名と対応させるため、DataFrameの列名は重複のない文字列にしてください。

```python
import pandas as pd

from malchan.app import dataframe_to_records


train_df = pd.DataFrame(
    {
        "x1": [0.1, 0.2, 0.3, 0.4],
        "x2": [1.0, 0.9, 0.7, 0.4],
        "y": [10.0, 12.0, 13.5, 16.0],
    }
)

train_payload = {
    "data": dataframe_to_records(train_df),
    "target_col": "y",
    "task": "regression",
    "num_cols": ["x1", "x2"],
    "cat_cols": [],
    "model_names": ["線形回帰"],
    "compute_xai": False,
}

train_response = client.post("/api/models", json=train_payload)
train_response.raise_for_status()
model_id = train_response.json()["model_id"]

predict_df = pd.DataFrame(
    {
        "x1": [0.25, 0.35],
        "x2": [0.8, 0.5],
    }
)

predict_response = client.post(
    f"/api/models/{model_id}/predict",
    json={"data": dataframe_to_records(predict_df)},
)
predict_response.raise_for_status()

prediction_df = pd.DataFrame(predict_response.json()["predictions"])
display(prediction_df)
```

欠損値や日時列を含まない単純なDataFrameでは、`df.to_dict(orient="records")`でも送信できます。ただし、Notebookで扱う実データにはpandas固有型が含まれることが多いため、通常は`dataframe_to_records()`を推奨します。

## NotebookでXAIを利用

XAIを利用する場合は、SHAPとPlotly可視化の依存関係もインストールします。

```bash
pip install -e ".[api,notebook,models,visualization]"
```

FastAPIのXAIエンドポイントはJSONを返します。Notebookでは`malchan.visualization`の次の関数へ、`response`または`response.json()`を直接渡して可視化できます。

```python
from malchan.visualization import (
    show_xai_importance,
    show_xai_pd_and_ice,
    show_xai_shap_scatter,
)
```

| 関数 | 対応するエンドポイント | 出力 |
|---|---|---|
| `show_xai_importance()` | `GET .../importance` | Plotly特徴量重要度棒グラフ |
| `show_xai_shap_scatter()` | `GET .../shap` | Plotly SHAP散布図 |
| `show_xai_pd_and_ice()` | `GET .../pdp` | Plotly PDP/ICE図 |

これらはFastAPIレスポンス用のアダプターです。学習済みPythonモデルを直接保持している場合は、従来の`show_importances()`、`show_shap_scatter()`、`show_pd_and_ice()`も利用できます。

### 1. XAIを有効にしてモデルを学習

`compute_xai=True`を指定します。既定値も`True`ですが、Notebookでは計算の有無を明示することを推奨します。

```python
feature_cols = ["x1", "x2"]
target_col = "y"

xai_train_response = client.post(
    "/api/models",
    json={
        "data": dataframe_to_records(train_df),
        "target_col": target_col,
        "task": "regression",
        "num_cols": feature_cols,
        "cat_cols": [],
        "model_names": ["ランダムフォレスト回帰"],
        "compute_xai": True,
    },
)
xai_train_response.raise_for_status()

model_info = xai_train_response.json()
model_id = model_info["model_id"]
print("model_id:", model_id)
print("xai_status:", model_info["xai_status"])
```

学習とXAI計算は現在同期処理です。`POST /api/models`の応答が返った時点で、成功した目的変数のキャッシュは利用可能です。XAI計算に失敗してもモデル登録自体は成功し、失敗理由はXAIサマリーに保存されます。

### 2. XAIサマリーを確認

最初にサマリーを取得し、利用可能な目的変数、特徴量、重要度手法を確認します。

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
```

目的変数ごとの情報には次が含まれます。

| Field | 内容 |
|---|---|
| `status` | `ready`、`failed`、`not_requested`などの状態 |
| `computed_at` | XAIキャッシュを計算した日時 |
| `error` | 計算に失敗した場合の例外情報 |
| `features` | モデルの入力特徴量 |
| `importance_methods` | 利用可能な`model`、`pfi`、`shap` |
| `shap_features` | SHAP散布データを取得できる特徴量 |
| `pdp_features` | PDP/ICEを取得できる特徴量 |

単一目的の場合も複数目的の場合も、目的変数は`targets`のキーから選択できます。

```python
target = next(
    name
    for name, info in xai_summary["targets"].items()
    if info["status"] == "ready"
)
target_info = xai_summary["targets"][target]
print("selected target:", target)
```

### 3. 特徴量重要度を取得して可視化

`method`には`model`、`pfi`、`shap`を指定できます。実際に利用可能な手法は`importance_methods`で確認してください。

```python
importance_method = "shap"

importance_response = client.get(
    f"/api/models/{model_id}/xai/{target}/importance",
    params={
        "method": importance_method,
        "combined": True,
        "top_n": 20,
    },
)
importance_response.raise_for_status()

importance_fig = show_xai_importance(
    importance_response,
    n_bar=20,
)
importance_fig.show()
```

`show_xai_importance()`はHTTPレスポンスと辞書の両方を受け取れます。表形式でも確認する場合は同じレスポンスをDataFrameへ変換します。

```python
importance_payload = importance_response.json()
importance_df = pd.DataFrame(importance_payload["items"])
display(importance_df)
print("combined:", importance_payload["combined"])
```

`combined=True`では、one-hot encodingや材料特徴量生成後の多数の列を、可能な場合は元の入力列単位へ集約します。対応する集約キャッシュがない場合は、自動的に前処理後の特徴量単位へフォールバックし、応答の`combined`が`False`になります。

### 4. 特徴量別SHAPデータを取得して可視化

SHAPエンドポイントは、1つの元特徴量について散布図を作るためのレコードを返します。利用可能な特徴量は`shap_features`から選択します。

```python
shap_feature = target_info["shap_features"][0]

shap_response = client.get(
    f"/api/models/{model_id}/xai/{target}/shap",
    params={"feature": shap_feature},
)
shap_response.raise_for_status()

shap_fig = show_xai_shap_scatter(shap_response)
shap_fig.show()
```

回帰では通常1つのSHAP系列、分類ではクラスごとに複数系列が表示されます。分類で1クラスだけを表示する場合は、クラス名を`target_item`へ指定します。

```python
shap_fig = show_xai_shap_scatter(
    shap_response,
    target_item="OK",
)
shap_fig.show()
```

表形式でも確認できます。

```python
shap_payload = shap_response.json()
shap_df = pd.DataFrame(shap_payload["records"])
display(shap_df.head())
print("SHAP columns:", shap_payload["value_columns"])
```

カテゴリ特徴量の場合は、必要に応じてカテゴリ別平均も確認できます。

```python
category_shap_df = (
    shap_df.groupby(shap_feature, dropna=False)[shap_payload["value_columns"]]
    .mean()
    .reset_index()
)
display(category_shap_df)
```

### 5. PDPとICEを取得して可視化

PDPだけを取得する場合は`include_ice=False`、個別サンプルのICEも含める場合は`True`を指定します。利用可能な特徴量は`pdp_features`から選択します。

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

pdp_fig = show_xai_pd_and_ice(
    pdp_response,
    ice=True,
    max_ice=30,
)
pdp_fig.show()
```

回帰では通常1系列、分類ではクラスごとに複数系列が表示されます。分類で特定クラスだけを表示する場合は`series_name`を指定します。

```python
pdp_fig = show_xai_pd_and_ice(
    pdp_response,
    series_name="OK",
    ice=True,
)
pdp_fig.show()
```

PDPだけ表示したい場合は、APIでICEを取得しないか、可視化時に`ice=False`を指定します。

```python
pdp_only_fig = show_xai_pd_and_ice(
    pdp_response,
    ice=False,
)
pdp_only_fig.show()
```

レスポンスを表形式で確認する場合は、系列ごとにDataFrameへ変換できます。

```python
pdp_payload = pdp_response.json()
pdp_frames = []
for series in pdp_payload["series"]:
    frame = pd.DataFrame(
        {
            pdp_feature: pdp_payload["x_values"],
            "target": target,
            "series": series["name"],
            "pdp": series["pd_values"],
        }
    )
    pdp_frames.append(frame)

pdp_df = pd.concat(pdp_frames, ignore_index=True)
display(pdp_df)
```

### 6. XAIを明示的に再計算

`compute_xai=False`で学習した場合や、キャッシュを明示的に更新したい場合は再計算エンドポイントを利用します。

1つの目的変数だけ再計算する例です。

```python
recompute_response = client.post(
    f"/api/models/{model_id}/xai/recompute",
    json={"targets": [target]},
)
recompute_response.raise_for_status()
xai_summary = recompute_response.json()
print(xai_summary["status"])
```

全目的変数を再計算する場合は空のリストを指定します。

```python
recompute_response = client.post(
    f"/api/models/{model_id}/xai/recompute",
    json={"targets": []},
)
recompute_response.raise_for_status()
```

存在しない目的変数、重複した目的変数、空文字を指定するとHTTP `422`になります。

### 7. 複数目的モデルのXAIを順番に取得

複数目的でもAPIの構造は同じです。`targets`を走査し、`ready`の目的変数を可視化できます。

```python
for current_target, info in xai_summary["targets"].items():
    if info["status"] != "ready":
        print(current_target, "is not ready:", info["error"])
        continue

    methods = info["importance_methods"]
    if not methods:
        continue

    response = client.get(
        f"/api/models/{model_id}/xai/{current_target}/importance",
        params={
            "method": methods[0],
            "combined": True,
            "top_n": 20,
        },
    )
    response.raise_for_status()

    print(current_target)
    show_xai_importance(response, n_bar=20).show()
```

### 8. XAI取得時のHTTPステータス

| Status | 主な原因 |
|---|---|
| `200` | キャッシュを正常に取得 |
| `404` | `model_id`が存在しない |
| `409` | XAIを未計算、計算失敗、または要求したキャッシュがない |
| `422` | 目的変数、特徴量、手法、query parameterが不正 |

エラー内容をNotebookで確認する例です。

```python
response = client.get(
    f"/api/models/{model_id}/xai/{target}/importance",
    params={"method": "shap", "combined": True, "top_n": 20},
)

if response.status_code == 409:
    print("XAI cache is not ready:", response.json()["detail"])
else:
    response.raise_for_status()
```

### XAIキャッシュのライフサイクル

- `compute_xai=True`では、モデル学習直後にXAIを一度計算します。
- XAIのGET APIは保存済みキャッシュを返すだけで再計算しません。
- `POST /xai/recompute`を呼ぶと、選択した目的変数のキャッシュを再構築します。
- モデル比較後に`activate_best=True`で登録モデルを置き換えた場合、元のモデルでXAIが要求されていれば、新しいモデルのXAIも再計算します。
- `DELETE /api/models/{model_id}`でモデルを削除するとXAIキャッシュも削除されます。
- Notebookで`create_app()`または`TestClient`を作り直すと、以前のインメモリモデルとXAIキャッシュは引き継がれません。

処理後に明示的に閉じる場合:

```python
client.close()
```

通常のPythonモジュールから外部サーバーとして起動する場合も、同じファクトリを利用できます。

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

## React開発

SHAPとPDPの事前計算を利用するため、`visualization` extraもインストールします。

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

添付HTMLのワークフローをReactコンポーネントへ分割しています。

| Step | React画面 | 主な処理 |
|---|---|---|
| 1 | Data | CSV/XLSX読込、型推定、プレビュー、欠損・列統計 |
| 2 | Explore | ヒストグラム、散布図、相関ヒートマップ |
| 3 | Prepare | 単一・複数目的、回帰/分類、説明変数選択 |
| 4 | Model | 学習、候補比較、最良モデルチューニング、有効化 |
| 5 | Explain | Y-Y図、残差図、キャッシュ済み重要度・SHAP・PDP |
| 6 | Optimize | 任意条件予測、数値範囲・カテゴリ候補を使った逆解析 |
| 7 | Report | 分析レポート用プロンプト生成 |

## APIエンドポイント

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/health` | 稼働確認 |
| `POST` | `/api/models` | 単一・複数目的モデルの学習、XAI事前計算、登録 |
| `GET` | `/api/models` | 登録モデル一覧 |
| `GET` | `/api/models/{model_id}` | モデル情報と`xai_status` |
| `POST` | `/api/models/{model_id}/predict` | 予測・分類確率 |
| `POST` | `/api/models/{model_id}/compare` | 候補モデル比較と任意のベストモデルチューニング |
| `GET` | `/api/models/{model_id}/comparison` | 最新比較結果 |
| `POST` | `/api/models/{model_id}/comparison/tune-best` | 比較後の追加チューニング |
| `GET` | `/api/models/{model_id}/xai` | キャッシュ済みXAIの状態と利用可能データ |
| `GET` | `/api/models/{model_id}/xai/{target}/importance` | モデル/PFI/SHAP重要度 |
| `GET` | `/api/models/{model_id}/xai/{target}/shap` | 特徴量別SHAP散布用データ |
| `GET` | `/api/models/{model_id}/xai/{target}/pdp` | 特徴量別PDPと任意のICE |
| `POST` | `/api/models/{model_id}/xai/recompute` | 明示的なXAI再計算 |
| `POST` | `/api/models/{model_id}/inverse-analysis` | Optuna逆解析 |
| `DELETE` | `/api/models/{model_id}` | 登録モデルとXAIキャッシュ削除 |

ReactのModel画面では`activate_best=true`を指定でき、比較・チューニング後のベストモデルを後続の予測と逆解析へ反映します。XAIが有効なモデルでは、ベストモデルを有効化した直後に新しいモデルのSHAP/PDPキャッシュも更新します。

## XAIの計算とHTTP API

`POST /api/models`の`compute_xai`は既定で`true`です。モデルの学習後に、各目的変数の子モデルで次を実行します。

```python
model.shap()
model.get_xai()
```

`get_xai()`が作成した`model.importances`には、モデル重要度、Permutation Importance、SHAP重要度、特徴量別SHAPデータ、特徴量別PDP/ICEが保存されます。

通常のXAI GETエンドポイントは`shap()`や`get_xai()`を呼ばず、保存済みキャッシュをシリアライズするだけです。Notebookでの詳細な取得・可視化方法は「NotebookでXAIを利用」を参照してください。

### curlでの利用例

状態と利用可能な目的変数・特徴量・手法:

```bash
curl http://127.0.0.1:8000/api/models/<model_id>/xai
```

SHAP重要度:

```bash
curl "http://127.0.0.1:8000/api/models/<model_id>/xai/y/importance?method=shap&combined=true&top_n=20"
```

SHAP散布用データ:

```bash
curl "http://127.0.0.1:8000/api/models/<model_id>/xai/y/shap?feature=x1"
```

PDPのみ:

```bash
curl "http://127.0.0.1:8000/api/models/<model_id>/xai/y/pdp?feature=x1"
```

最大30サンプルのICE曲線も含める場合:

```bash
curl "http://127.0.0.1:8000/api/models/<model_id>/xai/y/pdp?feature=x1&include_ice=true&max_ice=30"
```

明示的に一部の目的変数を再計算する場合:

```bash
curl -X POST http://127.0.0.1:8000/api/models/<model_id>/xai/recompute \
  -H "Content-Type: application/json" \
  -d '{"targets": ["strength"]}'
```

## XAI状態

モデル全体と目的変数ごとに次の状態を返します。

- `not_requested`: 学習時に無効化され、まだ計算していない
- `computing`: 計算中
- `ready`: キャッシュが利用可能
- `partial`: 複数目的のうち一部だけ成功
- `failed`: XAI計算に失敗
- `unavailable`: モデルが`shap()`または`get_xai()`を提供していない

XAI計算で例外が発生してもモデル登録自体は成功します。詳細は目的変数別の`error`に記録されます。

## Web配信設定

`create_app()`はAPIルートを登録した後、次の順番でReactビルドを探索します。

1. `MALCHAN_FRONTEND_DIST`
2. `src/malchan/app/web/static`
3. 開発用の`frontend/dist`

`index.html`が見つかった場合だけ`/`へ静的マウントします。API専用運用では:

```bash
MALCHAN_SERVE_FRONTEND=false uvicorn "malchan.app:create_app" --factory
```

Vite開発サーバーのCORS許可は既定で次の2つです。

- `http://127.0.0.1:5173`
- `http://localhost:5173`

変更する場合:

```bash
MALCHAN_CORS_ORIGINS="http://localhost:5173,http://example.local" \
  uvicorn "malchan.app:create_app" --factory
```

## 現在の制約

- 学習、XAI事前計算、比較、チューニング、逆解析は同期処理です。
- XAI事前計算は全特徴量のPDP/ICEを保持するため、特徴量数・データ数に応じて学習時間とメモリ使用量が増えます。
- モデル、比較状態、XAIキャッシュはプロセス内メモリに保持され、複数workerでは共有されません。
- Reactはデータをブラウザで解析してJSON送信します。大規模ファイル向けのストリーミングアップロードは未実装です。
- 元HTMLにある固定値・線形制約などの高度な逆解析UIは、既存APIスキーマへ段階的に接続する予定です。
