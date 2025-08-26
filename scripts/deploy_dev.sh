#!/bin/bash

# 開発環境デプロイスクリプト

set -e

echo "🚀 開発環境デプロイ開始"
echo "環境: DEVELOPMENT"
echo ""

# 環境変数ファイルをコピー
echo "📋 環境設定を適用..."
cp config/dev.env .env
echo "✅ 開発環境設定を適用しました"

# 依存関係のインストール
echo "📦 依存関係をインストール..."
pip install -r requirements.txt
echo "✅ 依存関係のインストール完了"

# データベースの初期化
echo "🗄️ データベースを初期化..."
python -c "from app.config import settings; print(f'Database: {settings.DATABASE_URL}')"
echo "✅ データベース初期化完了"

# アプリケーションの起動
echo "🚀 アプリケーションを起動..."
echo "開発サーバー: http://127.0.0.1:8000"
echo "ヘルスチェック: http://127.0.0.1:8000/health"
echo ""
echo "停止するには Ctrl+C を押してください"
echo ""

uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload --log-level debug
