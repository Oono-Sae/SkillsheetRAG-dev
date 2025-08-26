from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    # アプリケーション設定
    APP_NAME: str = "Skillsheet RAG System"
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    
    # サーバー設定
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # ファイル設定
    UPLOAD_DIR: str = "uploads"
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB
    ALLOWED_EXTENSIONS: list = [".xlsx", ".pdf"]
    
    # データベース設定
    DATABASE_URL: str = "sqlite:///./skillsheet.db"
    
    # RAG設定
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"


    
    # OpenAI GPT設定
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o"  # GPT-4.5 (gpt-4o) を使用
    OPENAI_MAX_TOKENS: int = 2000  # GPT-4.5用にトークン数を増加
    OPENAI_TEMPERATURE: float = 0.3  # より正確な回答のため温度を下げる
    OPENAI_TOP_P: float = 0.9  # トップPパラメータを追加
    OPENAI_FREQUENCY_PENALTY: float = 0.1  # 頻度ペナルティを追加
    OPENAI_PRESENCE_PENALTY: float = 0.1  # 存在ペナルティを追加
    
    # Redis設定
    REDIS_URL: str = "redis://localhost:6379"
    
    # セキュリティ設定
    SECRET_KEY: str = "your-secret-key-here"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# 設定インスタンス
settings = Settings()

# 環境変数から設定を読み込み
if os.getenv("ENVIRONMENT"):
    settings.ENVIRONMENT = os.getenv("ENVIRONMENT")
if os.getenv("DATABASE_URL"):
    settings.DATABASE_URL = os.getenv("DATABASE_URL")
if os.getenv("REDIS_URL"):
    settings.REDIS_URL = os.getenv("REDIS_URL")
if os.getenv("SECRET_KEY"):
    settings.SECRET_KEY = os.getenv("SECRET_KEY")
