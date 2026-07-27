# malchan React workbench

添付のスタンドアロンHTMLを参考に、malchanのFastAPI APIへ接続するReactワークベンチです。

## 開発起動

XAIの事前計算とWeb用Plotly図の生成には`visualization` extraが必要です。

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

ブラウザで `http://127.0.0.1:5173` を開きます。Viteは`/api`を`http://127.0.0.1:8000`へプロキシします。

## 本番ビルド

```bash
cd frontend
npm install
npm run build
cd ..
uvicorn "malchan.app:create_app" --factory
```

Viteの出力先は`src/malchan/app/web/static`です。FastAPIはビルド済みのReactアプリを`/`から配信します。

## 実装済みフロー

1. Data: CSV/XLSX読込、型推定、表プレビュー、欠損・列統計
2. Explore: ヒストグラム、散布図、相関ヒートマップ
3. Prepare: 単一・複数目的、回帰/分類、説明変数選択
4. Model: 学習前の候補比較、通常学習、ベストモデルだけのOptunaチューニング、有効化
5. Explain: Y-Y図、特徴量重要度、SHAP Beeswarm、1D/2D Partial Dependenceを縦並びで表示
6. Optimize: 任意条件予測、数値範囲・カテゴリ候補を使った逆解析
7. Report: データ・比較・逆解析結果を含むレポート用プロンプト

Model画面で登録モデルがない状態から「未学習から比較」を実行すると、Webアプリが現在の学習設定で比較用モデルをFastAPIへ登録し、続けて登録モデル用の比較APIを呼び出します。これにより、利用者が単独学習を先に実行しなくても候補比較を開始できます。

## Plotly図とvisualization

Explain画面では、React側でグラフ形状を独自実装しません。FastAPIが`malchan.visualization`の関数を呼び出し、生成された`plotly.graph_objects.Figure`をPlotly JSONとして返します。Reactは返された`data`と`layout`を`plotly.js-dist-min`で描画します。

利用する主な可視化関数は次のとおりです。

- Y-Y／混同行列: `show_model_diagnostics` → `yy_plot_ml`
- 特徴量重要度: `show_xai_importance`
- SHAP Beeswarm: `show_xai_shap_beeswarm`
- 1D PD／ICE: `show_xai_pd_and_ice`
- 2D PD: `show_model_pd_2d`

この構成により、Notebook・Pythonコード・FastAPI Web画面で可視化ロジックを共通化します。

## XAIの事前計算とキャッシュ

モデル学習APIの`compute_xai`は既定で`true`です。学習完了後に各目的変数のモデルで次を一度だけ実行します。

```python
model.shap()
model.get_xai()
```

計算されたモデル重要度、Permutation Importance、SHAP重要度、全SHAP値、PDP/ICEデータは学習済みモデルに保持されます。Explain画面で目的変数・特徴量・重要度手法を切り替えても、通常のGET APIはキャッシュを読むだけで再計算しません。

計算時間を避けたい場合は学習リクエストで無効化できます。

```json
{
  "compute_xai": false
}
```

無効化後や明示的に更新したい場合だけ、Explain画面の「XAIを再計算」または`POST /xai/recompute`を使います。状態確認用の`GET /xai`は未計算時も利用できますが、重要度・SHAP・1D PDPの取得はHTTP `409`になります。

モデル比較・チューニングで`activate_best=true`を指定した場合、登録モデルの置換後に、計算要求済みのXAIキャッシュを新しいベストモデルへ更新します。

## Visualization API

| Method | Path | visualization出力 |
|---|---|---|
| `GET` | `/api/models/{model_id}/visualizations/{target}/yy` | Y-Y図または混同行列 |
| `GET` | `/api/models/{model_id}/visualizations/{target}/importance` | 重要度棒グラフ |
| `GET` | `/api/models/{model_id}/visualizations/{target}/shap-beeswarm` | SHAP Beeswarm |
| `GET` | `/api/models/{model_id}/visualizations/{target}/pdp` | 1D PDP／ICE |
| `GET` | `/api/models/{model_id}/visualizations/{target}/pdp-2d` | 2D PDP等高線 |

各レスポンスは次の形式です。

```json
{
  "figure": {
    "data": [],
    "layout": {}
  },
  "metadata": {}
}
```

2D PDはブラウザで予測グリッドを作らず、FastAPI側で学習済みモデルの`get_pd_2d()`を呼び、`malchan.visualization`が等高線図を生成します。

## XAI API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/models/{model_id}/xai` | キャッシュ状態、利用可能な目的変数・特徴量・手法 |
| `GET` | `/api/models/{model_id}/xai/{target}/importance` | モデル/PFI/SHAP重要度 |
| `GET` | `/api/models/{model_id}/xai/{target}/shap-values` | Beeswarm用の全特徴量SHAP行列 |
| `GET` | `/api/models/{model_id}/xai/{target}/shap?feature=x1` | 特徴量単位のSHAP散布用レコード |
| `GET` | `/api/models/{model_id}/xai/{target}/pdp?feature=x1` | 1D PDPと任意のICE曲線 |
| `POST` | `/api/models/{model_id}/xai/recompute` | 明示的なXAI再計算 |

## 環境変数

- `VITE_API_BASE`: Viteから利用するAPIベース。既定値は`/api`
- `MALCHAN_CORS_ORIGINS`: 許可するOriginのカンマ区切り
- `MALCHAN_SERVE_FRONTEND`: `false`でFastAPIの静的配信を無効化
- `MALCHAN_FRONTEND_DIST`: Reactビルドディレクトリを明示指定

## 現在の制約

- CSV/XLSXはブラウザで読み込み、学習時にJSONとしてFastAPIへ送信します。大規模データ向けのアップロードAPIは未実装です。
- XAI事前計算は同期処理です。特徴量数・データ数・モデル種別によって学習APIの応答時間とメモリ使用量が増えます。
- 2D PDも同期処理です。既定の可視化関数では特徴量ごとのグリッドと最大300件の学習サンプルを使用するため、モデルによっては応答に時間がかかります。
- XAI計算の一部が失敗してもモデル登録は成功し、`xai_status`と目的変数別の`error`に状態を記録します。
- 元HTMLにある固定値・線形制約などの高度な逆解析UIは、既存FastAPIスキーマに合わせて次段階で拡張できます。
- モデル、比較結果、XAIキャッシュは現在プロセス内メモリに保存されます。
