# malchan FastAPI + React application

`malchan.app`は、モデル学習・予測・比較・ベストモデルチューニング・XAI・逆解析を提供するFastAPIと、スタンドアロンHTMLを参考にしたReactワークベンチをまとめたアプリケーション層です。

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

## XAIをモデル作成後に一度だけ計算する

`POST /api/models`の`compute_xai`は既定で`true`です。モデルの学習後に、各目的変数の子モデルで次を実行します。

```python
model.shap()
model.get_xai()
```

`get_xai()`が作成した`model.importances`には、モデル重要度、Permutation Importance、SHAP重要度、特徴量別SHAPデータ、特徴量別PDP/ICEが保存されます。

```json
{
  "data": [
    {"x1": 0.1, "x2": 1.0, "y": 10.0},
    {"x1": 0.2, "x2": 0.9, "y": 12.0}
  ],
  "target_col": "y",
  "task": "regression",
  "num_cols": ["x1", "x2"],
  "cat_cols": [],
  "model_names": ["ランダムフォレスト回帰"],
  "compute_xai": true
}
```

通常のXAI GETエンドポイントは`shap()`や`get_xai()`を呼ばず、保存済みキャッシュをシリアライズするだけです。このため、Reactで目的変数、特徴量、重要度手法を切り替えても再計算しません。

XAIを学習時に計算しない場合:

```json
{
  "compute_xai": false
}
```

この場合、`xai_status`は`not_requested`になり、重要度・SHAP・PDP取得はHTTP `409`を返します。状態確認用の`GET /xai`は常に利用できます。

## XAI APIの利用例

状態と利用可能な目的変数・特徴量・手法:

```bash
curl http://127.0.0.1:8000/api/models/<model_id>/xai
```

SHAP重要度:

```bash
curl "http://127.0.0.1:8000/api/models/<model_id>/xai/y/importance?method=shap&combined=true&top_n=20"
```

SHAP dependence用データ:

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

再計算を行うのは、このPOSTまたは`activate_best=true`による登録モデル置換時だけです。概要・重要度・SHAP・PDPのGETを繰り返しても、計算回数は増えません。

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
