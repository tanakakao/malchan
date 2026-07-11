# malchan React workbench

添付のスタンドアロンHTMLを参考に、malchanのFastAPI APIへ接続するReactワークベンチです。

## 開発起動

XAIの事前計算にはSHAPを含む`visualization` extraが必要です。

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
4. Model: 学習、モデル比較、ベストモデルだけのOptunaチューニング、有効化
5. Explain: Y-Y図、残差図、キャッシュ済みSHAP重要度・SHAP dependence・PDP
6. Optimize: 任意条件予測、数値範囲・カテゴリ候補を使った逆解析
7. Report: データ・比較・逆解析結果を含むレポート用プロンプト

## XAIの事前計算とキャッシュ

モデル学習APIの`compute_xai`は既定で`true`です。学習完了後に各目的変数のモデルで次を一度だけ実行します。

```python
model.shap()
model.get_xai()
```

計算されたモデル重要度、Permutation Importance、SHAP重要度、SHAP散布用データ、PDP/ICEデータは学習済みモデルに保持されます。Explain画面で目的変数・特徴量・重要度手法を切り替えても、通常のGET APIはキャッシュを読むだけで再計算しません。

計算時間を避けたい場合は学習リクエストで無効化できます。

```json
{
  "compute_xai": false
}
```

無効化後や明示的に更新したい場合だけ、Explain画面の「XAIを再計算」または`POST /xai/recompute`を使います。状態確認用の`GET /xai`は未計算時も利用できますが、重要度・SHAP・PDPの取得はHTTP `409`になります。

モデル比較・チューニングで`activate_best=true`を指定した場合、登録モデルの置換後に新しいベストモデルのXAIキャッシュを自動更新します。

## XAI API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/models/{model_id}/xai` | キャッシュ状態、利用可能な目的変数・特徴量・手法 |
| `GET` | `/api/models/{model_id}/xai/{target}/importance` | モデル/PFI/SHAP重要度 |
| `GET` | `/api/models/{model_id}/xai/{target}/shap?feature=x1` | SHAP散布用レコード |
| `GET` | `/api/models/{model_id}/xai/{target}/pdp?feature=x1` | PDPと任意のICE曲線 |
| `POST` | `/api/models/{model_id}/xai/recompute` | 明示的なXAI再計算 |

重要度の例:

```text
GET /api/models/<model_id>/xai/y/importance?method=shap&combined=true&top_n=20
```

PDPにICEを含める例:

```text
GET /api/models/<model_id>/xai/y/pdp?feature=temperature&include_ice=true&max_ice=30
```

選択した目的変数だけを再計算する例:

```json
{
  "targets": ["strength"]
}
```

通常の概要・重要度・SHAP・PDPのGETを繰り返しても、`shap()`と`get_xai()`の呼び出し回数は増えません。

## 環境変数

- `VITE_API_BASE`: Viteから利用するAPIベース。既定値は`/api`
- `MALCHAN_CORS_ORIGINS`: 許可するOriginのカンマ区切り
- `MALCHAN_SERVE_FRONTEND`: `false`でFastAPIの静的配信を無効化
- `MALCHAN_FRONTEND_DIST`: Reactビルドディレクトリを明示指定

## 現在の制約

- CSV/XLSXはブラウザで読み込み、学習時にJSONとしてFastAPIへ送信します。大規模データ向けのアップロードAPIは未実装です。
- XAI事前計算は同期処理です。特徴量数・データ数・モデル種別によって学習APIの応答時間とメモリ使用量が増えます。
- XAI計算の一部が失敗してもモデル登録は成功し、`xai_status`と目的変数別の`error`に状態を記録します。
- 元HTMLにある固定値・線形制約などの高度な逆解析UIは、既存FastAPIスキーマに合わせて次段階で拡張できます。
- モデル、比較結果、XAIキャッシュは現在プロセス内メモリに保存されます。
