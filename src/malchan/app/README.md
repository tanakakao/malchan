# malchan FastAPI + React application

`malchan.app`は、モデル学習・予測・比較・ベストモデルチューニング・逆解析を提供するFastAPIと、スタンドアロンHTMLを参考にしたReactワークベンチをまとめたアプリケーション層です。

## React開発

ターミナル1:

```bash
pip install -e ".[web,models,inverse,test]"
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
| 5 | Explain | Y-Y図、残差図、将来のSHAP/PDP領域 |
| 6 | Optimize | 任意条件予測、数値範囲・カテゴリ候補を使った逆解析 |
| 7 | Report | 分析レポート用プロンプト生成 |

## APIエンドポイント

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/health` | 稼働確認 |
| `POST` | `/api/models` | 単一・複数目的モデルの学習と登録 |
| `GET` | `/api/models` | 登録モデル一覧 |
| `GET` | `/api/models/{model_id}` | モデル情報 |
| `POST` | `/api/models/{model_id}/predict` | 予測・分類確率 |
| `POST` | `/api/models/{model_id}/compare` | 候補モデル比較と任意のベストモデルチューニング |
| `GET` | `/api/models/{model_id}/comparison` | 最新比較結果 |
| `POST` | `/api/models/{model_id}/comparison/tune-best` | 比較後の追加チューニング |
| `POST` | `/api/models/{model_id}/inverse-analysis` | Optuna逆解析 |
| `DELETE` | `/api/models/{model_id}` | 登録モデル削除 |

ReactのModel画面では`activate_best=true`を指定でき、比較・チューニング後のベストモデルを後続の予測と逆解析へ反映します。

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

- 学習、比較、チューニング、逆解析は同期処理です。
- モデルと比較状態はプロセス内メモリに保持され、複数workerでは共有されません。
- Reactはデータをブラウザで解析してJSON送信します。大規模ファイル向けのストリーミングアップロードは未実装です。
- Python側のSHAP、Permutation Importance、PDPをWebで利用するには説明可能性APIの追加が必要です。
- 元HTMLにある固定値・線形制約などの高度な逆解析UIは、既存APIスキーマへ段階的に接続する予定です。
