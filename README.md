# Skillsheet RAG System - 開発環境

## 📋 概要

スキルシートファイルをアップロードしてRAG検索ができるシステムの開発環境です。

## 🚀 クイックスタート

### 1. 環境の準備

```bash
# リポジトリをクローン
git clone https://github.com/Oono-Sae/skillsheet-rag-system-dev.git
cd skillsheet-rag-system-dev

# 仮想環境を作成
python -m venv venv
source venv/bin/activate  # Linux/Mac
# または
venv\Scripts\activate  # Windows
```

### 2. 依存関係のインストール

```bash
pip install -r requirements.txt
```

### 3. 環境設定

```bash
# 開発環境設定を適用
cp config/dev.env .env

# .envファイルを編集してAPIキーを設定
# OPENAI_API_KEY=your-actual-api-key
```

### 4. アプリケーションの起動

```bash
# デプロイスクリプトを使用
./scripts/deploy_dev.sh

# または手動で起動
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload --log-level debug
```

## 🌐 アクセス

- **アプリケーション**: http://127.0.0.1:8000
- **ヘルスチェック**: http://127.0.0.1:8000/health
- **API ドキュメント**: http://127.0.0.1:8000/docs

## 🔧 開発機能

### デバッグ機能
- 詳細なログ出力
- ホットリロード
- エラーの詳細表示

### テスト機能
- LLM単体テスト: `/smoke`
- 環境変数確認: `/debug/env`
- 各種LLMテスト: `/test/llm-*`

## 📁 ディレクトリ構造

```
skillsheet-rag-system-dev/
├── app/                    # アプリケーションコード
│   ├── main.py            # FastAPIアプリケーション
│   ├── config.py          # 設定管理
│   ├── models/            # データモデル
│   └── services/          # ビジネスロジック
├── frontend/              # フロントエンド
│   └── index.html         # メインUI
├── config/                # 環境設定
│   └── dev.env           # 開発環境設定
├── scripts/               # スクリプト
│   └── deploy_dev.sh     # 開発環境デプロイ
├── uploads/               # アップロードファイル
├── chroma_db/             # RAGデータベース
└── requirements.txt       # Python依存関係
```

## 🛠️ 開発コマンド

```bash
# サーバー起動
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

# テスト実行
python -m pytest tests/

# コードフォーマット
black app/

# リント
flake8 app/
```

## 🔍 トラブルシューティング

### よくある問題

1. **ポート8000が使用中**
   ```bash
   # 別のポートで起動
   uvicorn app.main:app --host 127.0.0.1 --port 8001 --reload
   ```

2. **OpenAI APIキーエラー**
   ```bash
   # .envファイルを確認
   cat .env | grep OPENAI_API_KEY
   ```

3. **依存関係エラー**
   ```bash
   # 仮想環境を再作成
   rm -rf venv
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

## 📝 開発ノート

- 開発環境では詳細なログが出力されます
- ファイルの変更は自動的にリロードされます
- デバッグ用のエンドポイントが利用可能です

## 🔗 関連リンク

- [ステージング環境](../skillsheet-rag-system-stg)
- [本番環境](../skillsheet-rag-system-prod)
- [メインプロジェクト](../skillsheet-rag-system)
