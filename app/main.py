from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Query
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os
from pathlib import Path
from typing import List, Optional
import logging
from dotenv import load_dotenv

# .envファイルの絶対パス指定でロード（作業ディレクトリ違いで未読になりがち）
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

from .services.file_service import FileService
from .services.rag_service import RAGService
from .services.gpt_service import GPTService
from .models.skillsheet import SkillsheetResponse, SearchResponse
from .config import settings

# ログ設定
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Skillsheet RAG System",
    description="スキルシートファイルをアップロードしてRAG検索ができるシステム（ローカルファイル + Google Docs対応）",
    version="1.0.0"
)

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# サービス初期化
file_service = FileService()
rag_service = RAGService()
gpt_service = GPTService()

@app.get("/")
async def root():
    """フロントエンドのHTMLファイルを配信"""
    try:
        # フロントエンドのHTMLファイルパス
        frontend_path = Path(__file__).parent.parent / "frontend" / "index.html"
        
        if frontend_path.exists():
            return FileResponse(frontend_path, media_type="text/html")
        else:
            # HTMLファイルが見つからない場合はAPI情報を返す
            return {"message": "Skillsheet RAG System API", "version": "1.0.0", "note": "Frontend HTML not found"}
    except Exception as e:
        logger.error(f"フロントエンド配信エラー: {str(e)}")
        return {"message": "Skillsheet RAG System API", "version": "1.0.0", "error": "Frontend delivery failed"}

@app.get("/health")
async def health_check():
    """ヘルスチェック"""
    return {"status": "healthy", "environment": settings.ENVIRONMENT}

@app.get("/smoke")
async def smoke_test():
    """LLM単体スモークテスト（RAGを外す）"""
    try:
        from openai import OpenAI, OpenAIError
        import os
        
        logger.debug("LLM単体テスト開始")
        
        # 環境変数の確認
        api_key = os.getenv("OPENAI_API_KEY")
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        
        logger.debug(f"APIキー: {api_key[:10] if api_key else 'None'}...")
        logger.debug(f"モデル: {model}")
        
        if not api_key:
            logger.error("OPENAI_API_KEYが設定されていません")
            raise HTTPException(status_code=400, detail="OPENAI_API_KEYが設定されていません")
        
        logger.debug("OpenAIクライアント初期化開始")
        # OpenAIクライアントを直接初期化（プロキシエラー回避）
        client = OpenAI(api_key=api_key)
        logger.debug("OpenAIクライアント初期化完了")
        
        logger.debug("GPT回答生成開始")
        # シンプルなテスト用の質問
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "say OK"}],
            max_tokens=10,
        )
        
        answer = response.choices[0].message.content
        logger.info(f"LLM単体テスト成功: {answer}")
        
        return {"answer": answer, "model": model, "status": "success"}
        
    except OpenAIError as e:
        # OpenAI API固有のエラー
        logger.error(f"OpenAI API エラー: {str(e)}")
        raise HTTPException(status_code=400, detail=f"OpenAI API エラー: {str(e)}")
    except Exception as e:
        logger.exception(f"LLM単体テスト失敗: {str(e)}")
        raise HTTPException(status_code=500, detail=f"server error: {str(e)}")

@app.get("/debug/env")
async def debug_env():
    """環境変数の確認（デバッグ用）"""
    import os
    return {
        "OPENAI_MODEL": os.getenv("OPENAI_MODEL"),
        "OPENAI_KEY_HEAD": (os.getenv("OPENAI_API_KEY") or "")[:10] + "..." if os.getenv("OPENAI_API_KEY") else "未設定",
        "ENVIRONMENT": os.getenv("ENVIRONMENT"),
        "CHROMA_PERSIST_DIR": os.getenv("CHROMA_PERSIST_DIR")
    }

@app.get("/test/llm-simple")
async def test_llm_simple():
    """LLM単体テスト（最小限のパラメータ）"""
    try:
        from openai import OpenAI
        import os
        
        logger.debug("LLM単体テスト（最小限）開始")
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=400, detail="OPENAI_API_KEYが設定されていません")
        
        # 最小限のパラメータでテスト
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # 最も軽量なモデル
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=5,
        )
        
        answer = response.choices[0].message.content
        logger.info(f"LLM単体テスト（最小限）成功: {answer}")
        
        return {"answer": answer, "model": "gpt-4o-mini", "status": "success"}
        
    except Exception as e:
        logger.exception(f"LLM単体テスト（最小限）失敗: {str(e)}")
        raise HTTPException(status_code=500, detail=f"test failed: {str(e)}")

@app.get("/test/llm-model")
async def test_llm_model():
    """LLM単体テスト（設定されたモデル）"""
    try:
        from openai import OpenAI
        import os
        
        logger.debug("LLM単体テスト（設定モデル）開始")
        
        api_key = os.getenv("OPENAI_API_KEY")
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        
        if not api_key:
            raise HTTPException(status_code=400, detail="OPENAI_API_KEYが設定されていません")
        
        # 設定されたモデルでテスト
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=5,
        )
        
        answer = response.choices[0].message.content
        logger.info(f"LLM単体テスト（設定モデル）成功: {answer}")
        
        return {"answer": answer, "model": model, "status": "success"}
        
    except Exception as e:
        logger.exception(f"LLM単体テスト（設定モデル）失敗: {str(e)}")
        raise HTTPException(status_code=500, detail=f"test failed: {str(e)}")

@app.get("/test/llm-clean")
async def test_llm_clean():
    """LLM単体テスト（環境変数完全クリア版）"""
    try:
        from openai import OpenAI
        import os
        
        logger.debug("LLM単体テスト（環境変数完全クリア）開始")
        
        # 現在の環境変数を保存
        original_env = dict(os.environ)
        
        # プロキシ関連の環境変数を完全にクリア
        proxy_vars = [
            'HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy',
            'NO_PROXY', 'no_proxy', 'ALL_PROXY', 'all_proxy',
            'HTTP_PROXY_USER', 'HTTPS_PROXY_USER', 'HTTP_PROXY_PASS', 'HTTPS_PROXY_PASS',
            'HTTP_PROXY_HOST', 'HTTPS_PROXY_HOST', 'HTTP_PROXY_PORT', 'HTTPS_PROXY_PORT',
            'HTTP_PROXY_AUTH', 'HTTPS_PROXY_AUTH'
        ]
        
        cleared_vars = {}
        for var in proxy_vars:
            if var in os.environ:
                cleared_vars[var] = os.environ[var]
                del os.environ[var]
                logger.debug(f"環境変数をクリア: {var}")
        
        # システムレベルのプロキシ関連環境変数もクリア
        for key in list(os.environ.keys()):
            if 'proxy' in key.lower() or 'PROXY' in key:
                if key not in cleared_vars:
                    cleared_vars[key] = os.environ[key]
                    del os.environ[key]
                    logger.debug(f"システムレベル環境変数をクリア: {key}")
        
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise HTTPException(status_code=400, detail="OPENAI_API_KEYが設定されていません")
            
            logger.debug("OpenAIクライアント初期化開始（環境変数クリア後）")
            # 環境変数クリア後にOpenAIクライアントを初期化
            client = OpenAI(api_key=api_key)
            logger.debug("OpenAIクライアント初期化完了")
            
            logger.debug("GPT回答生成開始")
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5,
            )
            
            answer = response.choices[0].message.content
            logger.info(f"LLM単体テスト（環境変数完全クリア）成功: {answer}")
            
            return {"answer": answer, "model": "gpt-4o-mini", "status": "success", "method": "environment_cleared"}
            
        finally:
            # 環境変数を復元
            for var, value in cleared_vars.items():
                os.environ[var] = value
                logger.debug(f"環境変数を復元: {var}")
        
    except Exception as e:
        logger.exception(f"LLM単体テスト（環境変数完全クリア）失敗: {str(e)}")
        raise HTTPException(status_code=500, detail=f"test failed: {str(e)}")

@app.get("/test/llm-subprocess")
async def test_llm_subprocess():
    """LLM単体テスト（サブプロセス完全分離版）"""
    try:
        import subprocess
        import sys
        import json
        
        logger.debug("LLM単体テスト（サブプロセス完全分離）開始")
        
        # サブプロセス用のPythonスクリプトを作成
        script_content = '''
import openai
import json
import os

try:
    # 環境変数を完全にクリア
    for key in list(os.environ.keys()):
        if 'proxy' in key.lower() or 'PROXY' in key:
            del os.environ[key]
    
    # APIキーを取得
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print(json.dumps({"status": "error", "message": "OPENAI_API_KEY not found"}))
        exit(1)
    
    # OpenAIクライアントを初期化
    client = openai.OpenAI(api_key=api_key)
    
    # テスト実行
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=5,
    )
    
    answer = response.choices[0].message.content
    print(json.dumps({"status": "success", "answer": answer, "model": "gpt-4o-mini"}))
    
except Exception as e:
    print(json.dumps({"status": "error", "message": str(e), "type": type(e).__name__}))
    exit(1)
'''
        
        # 一時的なPythonスクリプトファイルを作成
        script_path = "temp_llm_test.py"
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        try:
            logger.debug("サブプロセスでLLMテスト実行")
            # サブプロセスで実行
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                env={'OPENAI_API_KEY': os.getenv("OPENAI_API_KEY")}
            )
            
            logger.debug(f"サブプロセス終了コード: {result.returncode}")
            logger.debug(f"サブプロセス標準出力: {result.stdout}")
            logger.debug(f"サブプロセス標準エラー: {result.stderr}")
            
            # 結果を解析
            if result.returncode == 0:
                try:
                    output = json.loads(result.stdout.strip())
                    if output['status'] == 'success':
                        logger.info(f"LLM単体テスト（サブプロセス完全分離）成功: {output['answer']}")
                        return {
                            "answer": output['answer'], 
                            "model": output['model'], 
                            "status": "success", 
                            "method": "subprocess_isolated"
                        }
                    else:
                        raise Exception(f"サブプロセスでエラー: {output['message']}")
                except json.JSONDecodeError:
                    raise Exception(f"サブプロセス出力の解析に失敗: {result.stdout}")
            else:
                raise Exception(f"サブプロセスが失敗: {result.stderr}")
                
        finally:
            # 一時ファイルを削除
            if os.path.exists(script_path):
                os.remove(script_path)
                logger.debug("一時ファイルを削除")
        
    except Exception as e:
        logger.exception(f"LLM単体テスト（サブプロセス完全分離）失敗: {str(e)}")
        raise HTTPException(status_code=500, detail=f"test failed: {str(e)}")

@app.get("/test/llm-httpx")
async def test_llm_httpx():
    """LLM単体テスト（httpxクライアント直接制御版）"""
    try:
        from openai import OpenAI
        import httpx
        import os
        
        logger.debug("LLM単体テスト（httpxクライアント直接制御）開始")
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=400, detail="OPENAI_API_KEYが設定されていません")
        
        # httpxクライアントを直接制御
        http_client = httpx.Client(
            proxies=None,  # プロキシを明示的に無効化
            verify=True,
            timeout=30.0
        )
        
        # OpenAIクライアントをhttpxクライアント付きで初期化
        client = OpenAI(
            api_key=api_key,
            http_client=http_client
        )
        
        logger.debug("OpenAIクライアント初期化完了（httpx制御）")
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=5,
        )
        
        answer = response.choices[0].message.content
        logger.info(f"LLM単体テスト（httpxクライアント直接制御）成功: {answer}")
        
        return {"answer": answer, "model": "gpt-4o-mini", "status": "success", "method": "httpx_controlled"}
        
    except Exception as e:
        logger.exception(f"LLM単体テスト（httpxクライアント直接制御）失敗: {str(e)}")
        raise HTTPException(status_code=500, detail=f"test failed: {str(e)}")

@app.get("/test/llm-requests")
async def test_llm_requests():
    """LLM単体テスト（requestsライブラリ直接使用版）"""
    try:
        import requests
        import json
        import os
        
        logger.debug("LLM単体テスト（requestsライブラリ直接使用）開始")
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=400, detail="OPENAI_API_KEYが設定されていません")
        
        # OpenAI APIを直接呼び出し
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 5
        }
        
        logger.debug("OpenAI API直接呼び出し開始")
        response = requests.post(url, headers=headers, json=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            answer = result['choices'][0]['message']['content']
            logger.info(f"LLM単体テスト（requestsライブラリ直接使用）成功: {answer}")
            
            return {"answer": answer, "model": "gpt-4o-mini", "status": "success", "method": "requests_direct"}
        else:
            raise Exception(f"OpenAI API エラー: {response.status_code} - {response.text}")
        
    except Exception as e:
        logger.exception(f"LLM単体テスト（requestsライブラリ直接使用）失敗: {str(e)}")
        raise HTTPException(status_code=500, detail=f"test failed: {str(e)}")



@app.post("/upload", response_model=SkillsheetResponse)
async def upload_skillsheet(file: UploadFile = File(...)):
    """スキルシートファイルをアップロード"""
    try:
        # ファイル形式チェック
        if not file.filename.lower().endswith(('.xlsx', '.pdf')):
            raise HTTPException(
                status_code=400, 
                detail="サポートされているファイル形式は .xlsx と .pdf のみです"
            )
        
        # ファイル保存
        saved_path = await file_service.save_file(file)
        
        # RAGシステムに追加
        await rag_service.add_document(saved_path, file.filename)
        
        return SkillsheetResponse(
            filename=file.filename,
            file_path=str(saved_path),
            message="ファイルが正常にアップロードされ、RAGシステムに追加されました"
        )
        
    except Exception as e:
        logger.error(f"ファイルアップロードエラー: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/files", response_model=List[SkillsheetResponse])
async def list_files():
    """アップロードされたファイル一覧を取得"""
    try:
        files = await file_service.list_files()
        return files
    except Exception as e:
        logger.error(f"ファイル一覧取得エラー: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search", response_model=SearchResponse)
async def search_skillsheets(query: str = Form(...), n_results: int = Query(10)):
    """スキルシートを検索"""
    try:
        # n_resultsの型変換と検証
        try:
            n_results = int(n_results) if n_results is not None else 10
            if n_results < 1 or n_results > 100:
                raise HTTPException(
                    status_code=400,
                    detail="n_resultsは1から100の間である必要があります"
                )
        except (ValueError, TypeError):
            raise HTTPException(
                status_code=400,
                detail="n_resultsは有効な数値である必要があります"
            )
        
        results = await rag_service.search(query, n_results)
        return SearchResponse(
            query=query,
            results=results,
            total_results=len(results),
            message="検索が完了しました"
        )
    except Exception as e:
        logger.error(f"検索エラー: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/rag/collection-info")
async def get_rag_collection_info():
    """RAGコレクション情報を取得"""
    try:
        info = await rag_service.get_collection_info()
        return {"collection_info": info, "message": "コレクション情報を取得しました"}
    except Exception as e:
        logger.error(f"コレクション情報取得エラー: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rag/clear")
async def clear_rag_collection():
    """RAGコレクションをクリア"""
    try:
        success = await rag_service.clear_collection()
        if success:
            return {"message": "RAGコレクションがクリアされました"}
        else:
            raise HTTPException(status_code=500, detail="コレクションのクリアに失敗しました")
    except Exception as e:
        logger.error(f"コレクションクリアエラー: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/files/{filename}")
async def delete_file(filename: str):
    """ファイルを削除"""
    try:
        await file_service.delete_file(filename)
        await rag_service.remove_document(filename)
        return {"message": f"ファイル {filename} が削除されました"}
    except Exception as e:
        logger.error(f"ファイル削除エラー: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/gpt/generate-answer")
async def generate_gpt_answer(query: str = Form(...), n_results: int = Query(5)):
    """GPTを使用して質問に対する回答を生成"""
    try:
        logger.debug(f"GPT回答生成開始: query='{query}', n_results={n_results}")
        
        if not gpt_service.is_available():
            logger.warning("GPTサービスが利用できません")
            raise HTTPException(
                status_code=400,
                detail="GPTサービスが利用できません。OpenAI APIキーを設定してください。"
            )
        
        logger.debug("RAG検索を開始")
        # まずRAG検索で関連情報を取得
        search_results = await rag_service.search(query, n_results)
        logger.debug(f"RAG検索完了: {len(search_results)}件の結果")
        
        if not search_results:
            logger.info("検索結果がありません")
            return {
                "query": query,
                "answer": "申し訳ございませんが、質問に関連する情報が見つかりませんでした。",
                "context": [],
                "message": "関連情報なしで回答を生成しました"
            }
        
        logger.debug("GPT回答生成を開始")
        # GPTで回答を生成
        answer = await gpt_service.generate_answer(query, search_results)
        
        if not answer:
            logger.error("GPT回答の生成に失敗しました")
            raise HTTPException(
                status_code=500,
                detail="GPT回答の生成に失敗しました"
            )
        
        logger.info(f"GPT回答生成成功: {len(answer)}文字")
        return {
            "query": query,
            "answer": answer,
            "context": search_results,
            "message": "GPT回答を生成しました"
        }
        
    except HTTPException:
        # HTTPExceptionはそのまま再送出
        raise
    except Exception as e:
        logger.exception(f"GPT回答生成エラー: {str(e)}")
        raise HTTPException(status_code=500, detail=f"GPT回答生成エラー: {str(e)}")

@app.get("/gpt/status")
async def get_gpt_status():
    """GPTサービスの状態を確認"""
    return {
        "available": gpt_service.is_available(),
        "model": gpt_service.model if gpt_service.is_available() else None,
        "message": "GPTサービス状態を確認しました"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
