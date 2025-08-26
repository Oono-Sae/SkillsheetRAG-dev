from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

class SkillsheetResponse(BaseModel):
    """スキルシートレスポンスモデル"""
    filename: str
    file_path: str
    file_size: Optional[int] = None
    upload_date: Optional[datetime] = None
    message: str

class SearchResult(BaseModel):
    """検索結果モデル"""
    filename: str
    content: str
    score: float
    metadata: Optional[Dict[str, Any]] = None

class SearchResponse(BaseModel):
    """検索レスポンスモデル"""
    query: str
    results: List[SearchResult]
    total_results: int
    message: str

class FileInfo(BaseModel):
    """ファイル情報モデル"""
    id: int
    filename: str
    file_path: str
    file_size: int
    file_type: str
    upload_date: datetime
    processed: bool = False

class ProcessingStatus(BaseModel):
    """処理状況モデル"""
    filename: str
    status: str  # "pending", "processing", "completed", "failed"
    progress: float = 0.0
    message: Optional[str] = None
    created_at: datetime
    updated_at: datetime
