# malchan React workbench

添付のスタンドアロンHTMLを参考に、malchanのFastAPI APIへ接続するReactワークベンチです。

## 開発起動

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
5. Explain: 学習データ予測によるY-Y図・残差図
6. Optimize: 任意条件予測、数値範囲・カテゴリ候補を使った逆解析
7. Report: データ・比較・逆解析結果を含むレポート用プロンプト

## 環境変数

- `VITE_API_BASE`: Viteから利用するAPIベース。既定値は`/api`
- `MALCHAN_CORS_ORIGINS`: 許可するOriginのカンマ区切り
- `MALCHAN_SERVE_FRONTEND`: `false`でFastAPIの静的配信を無効化
- `MALCHAN_FRONTEND_DIST`: Reactビルドディレクトリを明示指定

## 現在の制約

- CSV/XLSXはブラウザで読み込み、学習時にJSONとしてFastAPIへ送信します。大規模データ向けのアップロードAPIは未実装です。
- SHAP、Permutation Importance、PDPはPython側に機能がありますが、HTTPエンドポイントは未実装です。Explain画面に拡張位置を確保しています。
- 元HTMLにある固定値・線形制約などの高度な逆解析UIは、既存FastAPIスキーマに合わせて次段階で拡張できます。
- モデルと比較結果は現在プロセス内メモリに保存されます。
