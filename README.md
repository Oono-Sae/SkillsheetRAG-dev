# Skillsheet RAG System - 本番環境

## 📋 概要

スキルシートファイルをアップロードしてRAG検索ができるシステムの本番環境です。

## 🚀 クイックスタート

### 1. 環境の準備

```bash
# リポジトリをクローン
git clone https://github.com/Oono-Sae/SkillsheetRAG-prod.git
cd SkillsheetRAG-prod

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
# 本番環境設定を適用
cp config/prod.env .env

# .envファイルを編集して本番用のAPIキーとシークレットを設定
# OPENAI_API_KEY=your-actual-production-api-key
# SECRET_KEY=your-actual-production-secret-key
```

### 4. アプリケーションの起動

```bash
# デプロイスクリプトを使用
./scripts/deploy_prod.sh

# または手動で起動
uvicorn app.main:app --host 0.0.0.0 --port 8000 --log-level warning
```

## 🌐 アクセス

- **アプリケーション**: http://0.0.0.0:8000
- **ヘルスチェック**: http://0.0.0.0:8000/health
- **API ドキュメント**: http://0.0.0.0:8000/docs

## 🔧 本番機能

### 運用機能
- 本番レベルのログ管理
- セキュリティ監査
- パフォーマンス監視
- エラー追跡

### ログ設定
- WARNINGレベルログ
- セキュリティログ
- アクセスログ
- エラーログ

## 📁 ディレクトリ構造

```
skillsheet-rag-system-prod/
├── app/                    # アプリケーションコード
│   ├── main.py            # FastAPIアプリケーション
│   ├── config.py          # 設定管理
│   ├── models/            # データモデル
│   └── services/          # ビジネスロジック
├── frontend/              # フロントエンド
│   └── index.html         # メインUI
├── config/                # 環境設定
│   └── prod.env          # 本番環境設定
├── scripts/               # スクリプト
│   └── deploy_prod.sh    # 本番環境デプロイ
├── nginx/                 # Nginx設定
│   └── nginx.prod.conf   # 本番用Nginx
├── uploads/               # アップロードファイル
├── chroma_db/             # RAGデータベース
└── requirements.txt       # Python依存関係
```

## 🛠️ 本番コマンド

```bash
# サーバー起動
uvicorn app.main:app --host 0.0.0.0 --port 8000 --log-level warning

# テスト実行
python -m pytest tests/

# Docker起動
docker-compose up -d
```

## 🔒 セキュリティ

### 本番環境のセキュリティ設定

- ✅ SSL/TLS暗号化
- ✅ セキュリティヘッダー
- ✅ レート制限
- ✅ 入力検証
- ✅ エラーハンドリング
- ✅ ログ監査

### 環境変数の管理

```bash
# 本番環境
cp config/prod.env .env
# 本番用のAPIキーとシークレットを設定
```

## 🔍 トラブルシューティング

### よくある問題

1. **ポート8000が使用中**
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8001 --log-level warning
   ```

2. **OpenAI APIキーエラー**
   ```bash
   # .envファイルを確認
   cat .env | grep OPENAI_API_KEY
   ```

3. **依存関係エラー**
   ```bash
   pip install -r requirements.txt --upgrade
   ```

## 📝 本番ノート

- 本番環境ではWARNINGレベルのログが出力されます
- セキュリティチェックが必須です
- 本番運用に適した設定が適用されています

## 🔗 関連リンク

- [開発環境](https://github.com/Oono-Sae/SkillsheetRAG-dev)
- [ステージング環境](https://github.com/Oono-Sae/SkillsheetRAG-stg)
- [メインプロジェクト](../skillsheet-rag-system)
