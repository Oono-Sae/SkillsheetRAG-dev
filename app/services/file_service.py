import os
import shutil
from pathlib import Path
from typing import List, Optional
from fastapi import UploadFile, HTTPException
import pandas as pd
import PyPDF2
import logging
from datetime import datetime

from ..config import settings
from ..models.skillsheet import SkillsheetResponse

logger = logging.getLogger(__name__)

class FileService:
    def __init__(self):
        self.upload_dir = Path(settings.UPLOAD_DIR)
        self.upload_dir.mkdir(exist_ok=True)
        
    async def save_file(self, file: UploadFile) -> Path:
        """ファイルを保存"""
        try:
            # ファイルサイズチェック（UploadFileは size を持たない場合があるため手動で計測）
            file_stream = getattr(file, "file", None)
            if file_stream and hasattr(file_stream, "seek") and hasattr(file_stream, "tell"):
                current_pos = file_stream.tell()
                file_stream.seek(0, os.SEEK_END)
                size_bytes = file_stream.tell()
                file_stream.seek(current_pos)
                if size_bytes > settings.MAX_FILE_SIZE:
                    raise HTTPException(
                        status_code=400,
                        detail=f"ファイルサイズが大きすぎます。最大{settings.MAX_FILE_SIZE // (1024*1024)}MBまで"
                    )
            
            # ファイル名の重複チェック
            filename = self._get_unique_filename(file.filename)
            file_path = self.upload_dir / filename
            
            # ファイル保存
            with open(file_path, "wb") as buffer:
                # 先ほどサイズ計測でポインタが進んでいる可能性があるため先頭へ
                if file_stream and hasattr(file_stream, "seek"):
                    file_stream.seek(0)
                shutil.copyfileobj(file.file, buffer)
            
            logger.info(f"ファイル保存完了: {filename}")
            return file_path
            
        except Exception as e:
            logger.error(f"ファイル保存エラー: {str(e)}")
            raise HTTPException(status_code=500, detail=f"ファイル保存に失敗しました: {str(e)}")
    
    def _get_unique_filename(self, filename: str) -> str:
        """重複しないファイル名を生成"""
        if not filename:
            filename = "unknown_file"
        
        name, ext = os.path.splitext(filename)
        counter = 1
        new_filename = filename
        
        while (self.upload_dir / new_filename).exists():
            new_filename = f"{name}_{counter}{ext}"
            counter += 1
        
        return new_filename
    
    async def list_files(self) -> List[SkillsheetResponse]:
        """アップロードされたファイル一覧を取得"""
        try:
            files = []
            for file_path in self.upload_dir.iterdir():
                if file_path.is_file():
                    stat = file_path.stat()
                    files.append(SkillsheetResponse(
                        filename=file_path.name,
                        file_path=str(file_path),
                        file_size=stat.st_size,
                        upload_date=datetime.fromtimestamp(stat.st_mtime),
                        message="ファイルが正常にアップロードされています"
                    ))
            return files
        except Exception as e:
            logger.error(f"ファイル一覧取得エラー: {str(e)}")
            raise HTTPException(status_code=500, detail=f"ファイル一覧取得に失敗しました: {str(e)}")
    
    async def delete_file(self, filename: str) -> bool:
        """ファイルを削除"""
        try:
            file_path = self.upload_dir / filename
            if not file_path.exists():
                raise HTTPException(status_code=404, detail="ファイルが見つかりません")
            
            file_path.unlink()
            logger.info(f"ファイル削除完了: {filename}")
            return True
            
        except Exception as e:
            logger.error(f"ファイル削除エラー: {str(e)}")
            raise HTTPException(status_code=500, detail=f"ファイル削除に失敗しました: {str(e)}")
    
    async def extract_text_from_excel(self, file_path: Path) -> str:
        """Excelファイルからテキストを抽出"""
        try:
            df = pd.read_excel(file_path, sheet_name=None)
            text_content = []
            
            for sheet_name, sheet_df in df.items():
                text_content.append(f"Sheet: {sheet_name}")
                text_content.append(sheet_df.to_string(index=False))
                text_content.append("")
            
            return "\n".join(text_content)
            
        except Exception as e:
            logger.error(f"Excelテキスト抽出エラー: {str(e)}")
            raise Exception(f"Excelファイルのテキスト抽出に失敗しました: {str(e)}")
    
    async def extract_text_from_pdf(self, file_path: Path) -> str:
        """PDFファイルからテキストを抽出"""
        try:
            text_content = []
            
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    text_content.append(f"Page {page_num + 1}")
                    text_content.append(page.extract_text())
                    text_content.append("")
            
            return "\n".join(text_content)
            
        except Exception as e:
            logger.error(f"PDFテキスト抽出エラー: {str(e)}")
            raise Exception(f"PDFファイルのテキスト抽出に失敗しました: {str(e)}")
    
    async def extract_text(self, file_path: Path) -> str:
        """ファイルからテキストを抽出（ファイル形式に応じて）"""
        try:
            if file_path.suffix.lower() == '.xlsx':
                return await self.extract_text_from_excel(file_path)
            elif file_path.suffix.lower() == '.pdf':
                return await self.extract_text_from_pdf(file_path)
            else:
                raise Exception(f"サポートされていないファイル形式: {file_path.suffix}")
                
        except Exception as e:
            logger.error(f"テキスト抽出エラー: {str(e)}")
            raise e
