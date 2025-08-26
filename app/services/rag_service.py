import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import asyncio
import gc

from ..config import settings
from ..services.file_service import FileService
from ..models.skillsheet import SearchResult

logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self):
        self.chroma_client = chromadb.PersistentClient(
            path=settings.CHROMA_PERSIST_DIR,
            settings=ChromaSettings(
                anonymized_telemetry=False
            )
        )
        
        # コレクション名
        self.collection_name = "skillsheets"
        
        # コレクションの取得または作成
        try:
            self.collection = self.chroma_client.get_collection(self.collection_name)
            logger.info(f"既存のコレクション '{self.collection_name}' を取得しました")
        except:
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"description": "スキルシートのRAG検索用コレクション"}
            )
            logger.info(f"新しいコレクション '{self.collection_name}' を作成しました")
        
        # ファイルサービス
        self.file_service = FileService()
        
        # 埋め込みモデルの遅延読み込み（軽量化）
        self._embedding_model = None
        self._model_loaded = False
        
        logger.info("RAGサービスを軽量化モードで初期化しました")
    
    @property
    def embedding_model(self):
        """埋め込みモデルを遅延読み込み"""
        if not self._model_loaded:
            logger.info(f"埋め込みモデル '{settings.EMBEDDING_MODEL}' を読み込み中...")
            self._embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
            self._model_loaded = True
            logger.info(f"埋め込みモデル '{settings.EMBEDDING_MODEL}' の読み込みが完了しました")
        return self._embedding_model
    
    def _clear_model_cache(self):
        """埋め込みモデルのキャッシュをクリアしてメモリを節約"""
        if self._model_loaded and self._embedding_model:
            del self._embedding_model
            self._embedding_model = None
            self._model_loaded = False
            gc.collect()  # ガベージコレクションを強制実行
            logger.info("埋め込みモデルのキャッシュをクリアしました")
    
    async def add_document(self, file_path: Path, filename: str) -> bool:
        """ドキュメントをRAGシステムに追加（軽量化版）"""
        try:
            # ファイルからテキストを抽出
            text_content = await self.file_service.extract_text(file_path)
            
            if not text_content.strip():
                logger.warning(f"ファイル '{filename}' からテキストが抽出できませんでした")
                return False
            
            # テキストをチャンクに分割
            chunks = self._split_text_into_chunks(text_content)
            
            # 埋め込みを一括計算（軽量化）
            embeddings = self.embedding_model.encode(chunks, convert_to_numpy=False, show_progress_bar=False)

            # 追加用データを構築
            ids = []
            metadatas = []
            for i, chunk in enumerate(chunks):
                ids.append(f"{filename}_chunk_{i}")
                metadatas.append({
                    "filename": filename,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "file_path": str(file_path),
                    "chunk_size": len(chunk)
                })

            # コレクションに一括追加（埋め込み付き）
            self.collection.add(
                documents=chunks,
                metadatas=metadatas,
                ids=ids,
                embeddings=[emb if isinstance(emb, list) else emb.tolist() for emb in embeddings]
            )
            
            # メモリ最適化
            del embeddings
            gc.collect()
            
            logger.info(f"ドキュメント '{filename}' をRAGシステムに追加しました（{len(chunks)}チャンク）")
            return True
            
        except Exception as e:
            logger.error(f"ドキュメント追加エラー '{filename}': {str(e)}")
            return False
    
    async def remove_document(self, filename: str) -> bool:
        """ドキュメントをRAGシステムから削除"""
        try:
            # メタデータで直接削除
            self.collection.delete(where={"filename": filename})
            logger.info(f"ドキュメント '{filename}' のチャンクを削除しました")
            
            return True
            
        except Exception as e:
            logger.error(f"ドキュメント削除エラー '{filename}': {str(e)}")
            return False
    
    async def search(self, query: str, n_results: int = 10) -> List[SearchResult]:
        """検索クエリでRAG検索を実行（軽量化版）"""
        try:
            # 埋め込みモデルが読み込まれていない場合は読み込み
            if not self._model_loaded:
                logger.info("検索実行のため埋め込みモデルを読み込み中...")
            
            # クエリの埋め込みを計算
            query_embedding = self.embedding_model.encode([query], convert_to_numpy=False)[0]
            
            # 検索実行
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
            
            # 結果をSearchResultオブジェクトに変換
            search_results = []
            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0], 
                    results['metadatas'][0], 
                    results['distances'][0]
                )):
                    # 類似度スコアを計算（距離を類似度に変換）
                    similarity_score = 1.0 / (1.0 + distance)
                    
                    search_results.append(SearchResult(
                        filename=metadata.get('filename', '不明なファイル'),
                        content=doc,
                        score=similarity_score,
                        metadata=metadata
                    ))
            
            # メモリ最適化
            del query_embedding
            gc.collect()
            
            logger.info(f"検索クエリ '{query}' で {len(search_results)} 件の結果を取得しました")
            return search_results
            
        except Exception as e:
            logger.error(f"検索エラー: {str(e)}")
            return []
    
    def _split_text_into_chunks(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """テキストをチャンクに分割（軽量化版）"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # チャンクの境界を調整（単語の境界で分割）
            if end < len(text):
                # 単語の境界を探す
                while end > start and text[end] != ' ' and text[end] != '\n':
                    end -= 1
                if end == start:  # 単語の境界が見つからない場合
                    end = start + chunk_size
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
            if start >= len(text):
                break
        
        return chunks
    
    async def get_collection_info(self) -> Dict[str, Any]:
        """コレクション情報を取得（軽量化版）"""
        try:
            count = self.collection.count()
            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "model_loaded": self._model_loaded,
                "memory_optimized": True
            }
        except Exception as e:
            logger.error(f"コレクション情報取得エラー: {str(e)}")
            return {"error": str(e)}
    
    def cleanup(self):
        """メモリクリーンアップ（軽量化）"""
        self._clear_model_cache()
        logger.info("RAGサービスのメモリクリーンアップが完了しました")
