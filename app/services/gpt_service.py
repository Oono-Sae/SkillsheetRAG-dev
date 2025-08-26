import openai
import logging
import os
import gc
from typing import List, Dict, Any, Optional
from ..config import settings

logger = logging.getLogger(__name__)

class GPTService:
    def __init__(self):
        self.api_key = settings.OPENAI_API_KEY
        self.model = settings.OPENAI_MODEL
        self.max_tokens = settings.OPENAI_MAX_TOKENS
        self.temperature = settings.OPENAI_TEMPERATURE
        self.top_p = getattr(settings, 'OPENAI_TOP_P', 0.9)
        self.frequency_penalty = getattr(settings, 'OPENAI_FREQUENCY_PENALTY', 0.1)
        self.presence_penalty = getattr(settings, 'OPENAI_PRESENCE_PENALTY', 0.1)
        
        if self.api_key:
            logger.info(f"GPT-4.5サービスを軽量化モードで初期化しました: {self.model}")
            logger.debug(f"APIキー: {self.api_key[:10]}...")
            logger.debug(f"モデル: {self.model}")
            logger.debug(f"最大トークン: {self.max_tokens}")
        else:
            logger.warning("OpenAI APIキーが設定されていません")
    
    def _get_openai_client(self):
        """OpenAI v1系に対応したクライアントを取得（プロキシ不要版）"""
        try:
            # パターンA：プロキシ不要（環境にPROXYが無い）
            client = openai.OpenAI(api_key=self.api_key)
            logger.info("OpenAIクライアント初期化成功（プロキシなし）")
            return client
        except Exception as e:
            logger.error(f"OpenAIクライアント初期化エラー: {str(e)}")
            raise e
    
    async def generate_answer(self, query: str, context: List[Dict[str, Any]]) -> Optional[str]:
        """GPTを使用して質問に対する回答を生成（軽量化版）"""
        logger.debug(f"generate_answer開始: query='{query}', context件数={len(context)}")
        
        if not self.api_key:
            logger.error("OpenAI APIキーが設定されていません")
            return None
        
        try:
            logger.debug("コンテキスト構築開始")
            # コンテキストを構築（軽量化）
            context_text = self._build_context_optimized(context)
            logger.debug(f"コンテキスト構築完了: {len(context_text)}文字")
            logger.debug(f"元のコンテキスト件数: {len(context)}件")
            logger.debug("メタデータ強化機能が有効化されています")
            
            # プロンプトを作成（GPT-4.5最適化版）
            system_prompt = """あなたはスキルシートの専門家であり、GPT-4.5レベルの高度な分析能力を持つAIアシスタントです。

与えられたコンテキストに基づいて、質問に対する詳細で正確な回答を提供してください。

【具体的な分析手順】
1. 各情報源の類似度スコアを確認し、信頼性を評価
2. ファイル名と内容の関連性を分析
3. 複数の情報源から得られる情報を統合
4. 具体的な数値や事実を抽出して提示
5. 情報の信頼性と制限について明確化

【回答の品質基準】
1. コンテキストに含まれる情報を積極的に活用し、可能な限り具体的な回答を提供する
2. ファイル名、類似度スコア、具体的な内容を参照して回答する
3. コンテキストの情報を組み合わせて、包括的な分析を行う
4. 具体的な例、数値、詳細を含める
5. 専門的で分かりやすい説明を心がける
6. 日本語で回答する
7. 論理的で構造化された回答を提供する
8. 必要に応じて箇条書きや表形式を使用する
9. コンテキストの信頼性や制限についても言及する

【回答の構造】
- 概要：質問に対する直接的な回答
- 詳細分析：コンテキストに基づく具体的な分析
- 根拠：回答の根拠となる情報源の明示
- 制限事項：情報の不足や信頼性について
- 推奨事項：追加情報が必要な場合の具体的な要求

コンテキストに情報がある場合は、その情報を最大限活用して回答してください。
コンテキストに情報がない場合のみ、「申し訳ございませんが、提供された情報からは回答できません。より詳細な情報が必要です。」と明記してください。"""

            user_prompt = f"""コンテキスト情報：
{context_text}

質問：{query}

上記のコンテキストに基づいて、質問に対する詳細な回答を提供してください。

【回答の要求事項】
1. コンテキストに含まれる具体的な情報（ファイル名、内容、類似度スコア）を積極的に参照する
2. 利用可能な情報を最大限活用して、具体的で有用な回答を提供する
3. 情報が不足している場合は、どのような追加情報が必要かを具体的に示す
4. 回答の根拠となるコンテキストの部分を明確に明示する
5. 類似度スコアの高い情報を優先的に活用する
6. 複数の情報源からの情報を統合して、包括的な分析を行う

【回答形式】
- 構造化された読みやすい形式で回答する
- 重要なポイントは箇条書きや表形式で整理する
- 情報の信頼性についても言及する
- 必要に応じて具体的な例や数値を提示する"""

            logger.debug("OpenAIクライアント初期化開始")
            # OpenAI v1系クライアントを取得
            client = self._get_openai_client()
            logger.debug("OpenAIクライアント初期化完了")
            
            logger.debug("GPT-4.5回答生成開始")
            # GPT-4.5回答生成
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty
            )
            
            answer = response.choices[0].message.content
            logger.info(f"GPT-4.5回答生成成功: {len(answer)}文字")
            logger.debug(f"回答内容: {answer[:100]}...")
            
            return answer
                
        except Exception as e:
            logger.exception(f"GPT回答生成エラー: {str(e)}")
            logger.error(f"エラーの詳細: {type(e).__name__}")
            return None
        
        finally:
            # メモリクリーンアップ
            gc.collect()
    
    def _build_context_optimized(self, context: List[Dict[str, Any]]) -> str:
        """コンテキストを最適化して構築（軽量化版）"""
        if not context:
            return "コンテキスト情報がありません。"
        
        # 最大5件までに制限（より多くの情報を提供）
        limited_context = context[:5]
        
        context_parts = []
        for i, item in enumerate(limited_context, 1):
            try:
                # 辞書形式のデータとして処理
                if isinstance(item, dict):
                    # メタデータを強化
                    enhanced_item = self._enhance_metadata(item, i)
                    
                    filename = enhanced_item['filename']
                    content = enhanced_item['content']
                    score = enhanced_item['score']
                    
                    # 内容の前処理を実行
                    content = self._preprocess_content(content)
                    
                    # 内容を800文字までに制限（より詳細な情報を提供）
                    if len(content) > 800:
                        content = content[:800] + "..."
                    
                    # 有効な内容のみを追加（強化されたメタデータ付き）
                    if content and content.strip():
                        metadata_summary = self._format_metadata_summary(enhanced_item)
                        context_parts.append(f"【情報{i}】\n{metadata_summary}\n内容:\n{content}")
                
                # オブジェクト形式のデータとして処理
                elif hasattr(item, 'filename') and hasattr(item, 'content'):
                    filename = getattr(item, 'filename', '不明なファイル')
                    content = getattr(item, 'content', '')
                    score = getattr(item, 'score', 0)
                    
                    # 内容の前処理を実行
                    content = self._preprocess_content(content)
                    
                    if len(content) > 800:
                        content = content[:800] + "..."
                    
                    if content and content.strip():
                        context_parts.append(f"【情報{i}】\nファイル: {filename}\n類似度スコア: {score:.3f}\n内容:\n{content}")
                
                # その他の形式
                else:
                    content = str(item)
                    # 内容の前処理を実行
                    content = self._preprocess_content(content)
                    
                    if len(content) > 800:
                        content = content[:800] + "..."
                    
                    if content and content.strip():
                        context_parts.append(f"【情報{i}】\n内容:\n{content}")
                
            except Exception as e:
                logger.warning(f"コンテキスト項目{i}の処理でエラー: {str(e)}")
                continue
        
        if not context_parts:
            return "有効なコンテキスト情報がありません。"
        
        # コンテキストの品質を検証・フィルタリング
        validated_parts = self._validate_context_quality(context_parts)
        
        if not validated_parts:
            return "品質基準を満たすコンテキスト情報がありません。"
        
        logger.debug(f"コンテキスト品質検証完了: {len(context_parts)}件 → {len(validated_parts)}件")
        
        return "\n\n".join(validated_parts)
    
    def _enhance_metadata(self, item: Dict[str, Any], index: int) -> Dict[str, Any]:
        """メタデータを強化して追加情報を提供"""
        enhanced_item = {}
        
        try:
            # 基本メタデータ
            enhanced_item['index'] = index
            enhanced_item['filename'] = item.get('filename', '不明なファイル')
            enhanced_item['score'] = item.get('score', 0)
            enhanced_item['content'] = item.get('content', '')
            
            # ファイル情報の強化
            if 'filename' in item:
                filename = item['filename']
                enhanced_item['file_extension'] = self._get_file_extension(filename)
                enhanced_item['file_type'] = self._get_file_type(filename)
                enhanced_item['file_size_category'] = self._get_file_size_category(item.get('file_size', 0))
            
            # 内容の分析情報
            if 'content' in item:
                content = item['content']
                enhanced_item['content_length'] = len(content) if content else 0
                enhanced_item['content_type'] = self._analyze_content_type(content)
                enhanced_item['key_entities'] = self._extract_key_entities(content)
                enhanced_item['language'] = self._detect_language(content)
            
            # 類似度スコアの詳細分析
            if 'score' in item:
                score = item['score']
                enhanced_item['relevance_level'] = self._get_relevance_level(score)
                enhanced_item['confidence_indicator'] = self._get_confidence_indicator(score)
            
            # タイムスタンプ情報
            enhanced_item['processed_at'] = self._get_current_timestamp()
            
        except Exception as e:
            logger.warning(f"メタデータ強化でエラー: {str(e)}")
            # エラーの場合は基本情報のみ
            enhanced_item = {
                'index': index,
                'filename': item.get('filename', '不明なファイル'),
                'score': item.get('score', 0),
                'content': item.get('content', ''),
                'error': 'メタデータ強化に失敗'
            }
        
        return enhanced_item
    
    def _get_file_extension(self, filename: str) -> str:
        """ファイル拡張子を取得"""
        try:
            if '.' in filename:
                return filename.split('.')[-1].lower()
            return '不明'
        except:
            return '不明'
    
    def _get_file_type(self, filename: str) -> str:
        """ファイルタイプを判定"""
        try:
            ext = self._get_file_extension(filename)
            if ext in ['xlsx', 'xls']:
                return 'Excel'
            elif ext == 'pdf':
                return 'PDF'
            elif ext in ['doc', 'docx']:
                return 'Word'
            elif ext in ['txt', 'csv']:
                return 'テキスト'
            else:
                return f'その他({ext})'
        except:
            return '不明'
    
    def _get_file_size_category(self, size_bytes: int) -> str:
        """ファイルサイズのカテゴリを判定"""
        try:
            if size_bytes == 0:
                return '不明'
            elif size_bytes < 1024:
                return '小(1KB未満)'
            elif size_bytes < 1024 * 1024:
                return '中(1MB未満)'
            else:
                return '大(1MB以上)'
        except:
            return '不明'
    
    def _analyze_content_type(self, content: str) -> str:
        """コンテンツのタイプを分析"""
        try:
            if not content:
                return '空'
            
            # 数値データの検出
            if any(char.isdigit() for char in content):
                if 'NaN' in content or 'nan' in content:
                    return '数値データ(一部無効)'
                else:
                    return '数値データ'
            
            # テキストデータの検出
            if len(content.split()) > 10:
                return '長文テキスト'
            elif len(content.split()) > 3:
                return '短文テキスト'
            else:
                return '短いテキスト'
                
        except:
            return '不明'
    
    def _extract_key_entities(self, content: str) -> str:
        """キーエンティティを抽出"""
        try:
            if not content or len(content) < 20:
                return '抽出不可'
            
            # 簡単なキーワード抽出（実際の実装ではより高度なNLPを使用）
            words = content.split()
            # 3文字以上の単語を抽出
            key_words = [word for word in words if len(word) >= 3 and word.isalnum()]
            # 上位5件を返す
            return ', '.join(key_words[:5]) if key_words else '抽出不可'
            
        except:
            return '抽出不可'
    
    def _detect_language(self, content: str) -> str:
        """言語を検出"""
        try:
            if not content:
                return '不明'
            
            # 日本語文字の検出
            japanese_chars = sum(1 for char in content if '\u3040' <= char <= '\u309f' or '\u30a0' <= char <= '\u30ff' or '\u4e00' <= char <= '\u9faf')
            total_chars = len(content)
            
            if japanese_chars > total_chars * 0.3:
                return '日本語'
            elif any('\u0000' <= char <= '\u007f' for char in content):
                return '英語/その他'
            else:
                return '不明'
                
        except:
            return '不明'
    
    def _get_relevance_level(self, score: float) -> str:
        """類似度スコアから関連性レベルを判定"""
        try:
            if score >= 0.8:
                return '非常に高い'
            elif score >= 0.6:
                return '高い'
            elif score >= 0.4:
                return '中程度'
            elif score >= 0.2:
                return '低い'
            else:
                return '非常に低い'
        except:
            return '不明'
    
    def _get_confidence_indicator(self, score: float) -> str:
        """類似度スコアから信頼度指標を判定"""
        try:
            if score >= 0.8:
                return '★★★★★ (最高)'
            elif score >= 0.6:
                return '★★★★☆ (高)'
            elif score >= 0.4:
                return '★★★☆☆ (中)'
            elif score >= 0.2:
                return '★★☆☆☆ (低)'
            else:
                return '★☆☆☆☆ (最低)'
        except:
            return '不明'
    
    def _get_current_timestamp(self) -> str:
        """現在のタイムスタンプを取得"""
        try:
            from datetime import datetime
            return datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        except:
            return '不明'
    
    def _format_metadata_summary(self, enhanced_item: Dict[str, Any]) -> str:
        """強化されたメタデータを読みやすい形式でフォーマット"""
        try:
            summary_parts = []
            
            # 基本情報
            summary_parts.append(f"ファイル: {enhanced_item['filename']}")
            summary_parts.append(f"類似度スコア: {enhanced_item['score']:.3f}")
            
            # ファイル情報
            if 'file_type' in enhanced_item:
                summary_parts.append(f"ファイルタイプ: {enhanced_item['file_type']}")
            if 'file_size_category' in enhanced_item:
                summary_parts.append(f"サイズ: {enhanced_item['file_size_category']}")
            
            # 内容分析
            if 'content_type' in enhanced_item:
                summary_parts.append(f"内容タイプ: {enhanced_item['content_type']}")
            if 'content_length' in enhanced_item:
                summary_parts.append(f"文字数: {enhanced_item['content_length']}")
            if 'language' in enhanced_item:
                summary_parts.append(f"言語: {enhanced_item['language']}")
            
            # 関連性と信頼度
            if 'relevance_level' in enhanced_item:
                summary_parts.append(f"関連性: {enhanced_item['relevance_level']}")
            if 'confidence_indicator' in enhanced_item:
                summary_parts.append(f"信頼度: {enhanced_item['confidence_indicator']}")
            
            # キーエンティティ
            if 'key_entities' in enhanced_item and enhanced_item['key_entities'] != '抽出不可':
                summary_parts.append(f"キーワード: {enhanced_item['key_entities']}")
            
            # 処理時刻
            if 'processed_at' in enhanced_item:
                summary_parts.append(f"処理時刻: {enhanced_item['processed_at']}")
            
            return '\n'.join(summary_parts)
            
        except Exception as e:
            logger.warning(f"メタデータサマリーフォーマットでエラー: {str(e)}")
            # エラーの場合は基本情報のみ
            return f"ファイル: {enhanced_item.get('filename', '不明')}\n類似度スコア: {enhanced_item.get('score', 0):.3f}"
    
    def _preprocess_content(self, content: str) -> str:
        """コンテンツの前処理（干渉なし）"""
        if not content:
            return ""
        
        try:
            # NaNの除去
            content = content.replace('NaN', '')
            content = content.replace('nan', '')
            content = content.replace('None', '')
            
            # 空白行の正規化
            lines = []
            for line in content.split('\n'):
                line = line.strip()
                if line and line not in ['', ' ', '\t']:
                    lines.append(line)
            
            content = '\n'.join(lines)
            
            # 特殊文字の処理
            content = content.replace('\t', ' ')
            content = content.replace('\r', '')
            
            # 連続する空白の正規化
            import re
            content = re.sub(r'\s+', ' ', content)
            
            # 空の括弧や無効な文字列の除去
            content = re.sub(r'\(\s*\)', '', content)
            content = re.sub(r'\[\s*\]', '', content)
            content = re.sub(r'\{\s*\}', '', content)
            
            # 内容の品質チェック
            if len(content.strip()) < 10:  # 10文字未満は無効
                return ""
            
            return content.strip()
            
        except Exception as e:
            logger.warning(f"コンテンツ前処理でエラー: {str(e)}")
            return content.strip() if content else ""
    
    def _validate_context_quality(self, context_parts: list) -> list:
        """コンテキストの品質を検証・フィルタリング"""
        if not context_parts:
            return []
        
        validated_parts = []
        for part in context_parts:
            try:
                # 内容の長さチェック
                if len(part) < 50:  # 50文字未満は除外
                    continue
                
                # 無効な内容のチェック
                if any(invalid in part.lower() for invalid in ['error', 'exception', 'traceback', 'stack']):
                    continue
                
                # 類似度スコアの妥当性チェック
                if '類似度スコア: 0.000' in part:  # 類似度0の情報は除外
                    continue
                
                validated_parts.append(part)
                
            except Exception as e:
                logger.warning(f"コンテキスト品質検証でエラー: {str(e)}")
                continue
        
        return validated_parts
    
    def is_available(self) -> bool:
        """GPTサービスが利用可能かチェック"""
        return bool(self.api_key)
    
    def cleanup(self):
        """リソースのクリーンアップ"""
        gc.collect()
        logger.info("GPTサービスリソースをクリーンアップしました")
