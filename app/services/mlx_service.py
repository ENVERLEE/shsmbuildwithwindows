from typing import Dict, List
from datetime import datetime
import json
import logging
import numpy as np
import os
import anthropic 
import voyageai
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

load_dotenv() 
XAI_API_KEY = os.getenv("XAI_API_KEY")


def setup_logging():
    """로깅 설정 개선"""
    logging.basicConfig(
        level=logging.INFO,  # DEBUG에서 INFO로 변경
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('mlx_service.log', encoding='utf-8')  # 인코딩 명시
        ]
    )

class MLXService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            
        try:
            self.anthropic = anthropic.Anthropic(api_key=XAI_API_KEY, base_url="https://api.x.ai",)
            self.voyage = voyageai.Client()
            self.model_name = "grok-beta"
            self._initialized = True
            self.logger.info("MLX Service initialized successfully")
        except Exception as e:
            self.logger.error(f"Initialization failed: {str(e)}")
            self._initialized = False
            raise
    def _extract_main_topics(self, text: str) -> List[str]:
        """텍스트에서 주요 토픽 추출"""
        try:
            from transformers import pipeline

            # 키워드 추출 파이프라인 설정 (한국어 BERT 모델 사용)
            extractor = pipeline(
                "token-classification",
                model="snunlp/KR-BERT-char16424",
                aggregation_strategy="simple"
            )
            
            # 텍스트 분석
            results = extractor(text)
            
            # 주요 엔티티 추출 (score가 0.5 이상인 것만)
            topics = [
                result['word'] 
                for result in results 
                if result['score'] > 0.5
            ]
            
            # 중복 제거 및 상위 10개만 선택
            unique_topics = list(dict.fromkeys(topics))[:10]
            
            return unique_topics

        except Exception as e:
            self.logger.error(f"Topic extraction failed: {str(e)}")
            return []
    def _safe_process_field(self, field_data: Dict) -> Dict:
        """필드 데이터 안전하게 처리"""
        try:
            if not isinstance(field_data, dict):
                return {"error": "Invalid field data format"}
            
            processed_data = {}
            for key, value in field_data.items():
                try:
                    # 데이터 검증
                    if isinstance(value, (str, int, float, bool)):
                        processed_data[key] = value
                    else:
                        processed_data[key] = str(value)
                except Exception as e:
                    self.logger.warning(f"Field processing error for key {key}: {str(e)}")
                    processed_data[key] = None
            
            return processed_data
        except Exception as e:
            self.logger.error(f"Field processing failed: {str(e)}")
            return {"error": str(e)}

    def process_request(self, data: Dict) -> Dict:
        """요청 처리 메서드"""
        if not self._initialized:
            return {"error": "Service not properly initialized"}
            
        try:
            # 데이터 유효성 검사
            if not data:
                raise ValueError("Empty request data")
                
            # 필드 데이터 처리
            processed_data = self._safe_process_field(data)
            if "error" in processed_data:
                raise ValueError(processed_data["error"])
                
            self.logger.info("Request processed successfully")
            return {
                "status": "success",
                "processed_data": processed_data,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            error_msg = f"Request processing failed: {str(e)}"
            self.logger.error(error_msg)
            return {
                "status": "error",
                "error": error_msg,
                "timestamp": datetime.utcnow().isoformat()
            }

    def health_check(self) -> Dict:
        """향상된 상태 체크"""
        try:
            # 기본 서비스 상태 확인
            status = {
                "initialized": self._initialized,
                "apis": {
                    "claude": "loaded" if hasattr(self, 'anthropic') else "not_loaded",
                    "voyage": "loaded" if hasattr(self, 'voyage') else "not_loaded"
                },
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # API 연결 테스트
            if self._initialized:
                try:
                    # 간단한 API 테스트
                    self.anthropic.messages.create(
                        model="grok-beta",
                        max_tokens=10,
                        messages=[{"role": "user", "content": "test"}]
                    )
                    status["apis"]["claude"] = "connected"
                except Exception as e:
                    status["apis"]["claude"] = f"error: {str(e)}"
                
                try:
                    self.voyage.embed(["test"], model="voyage-multilingual-2", input_type="document")
                    status["apis"]["voyage"] = "connected"
                except Exception as e:
                    status["apis"]["voyage"] = f"error: {str(e)}"
                
            status["status"] = "healthy" if self._initialized else "unhealthy"
            return status
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

# 메인 실행 코드
if __name__ == "__main__":
    try:
        setup_logging()
        logger = logging.getLogger(__name__)
        
        # 서비스 초기화 전 상태 확인
        logger.info("Initializing MLX Service...")
        
        service = MLXService()
        
        # 초기화 후 상태 확인
        health_status = service.health_check()
        logger.info(f"Service health status: {json.dumps(health_status, indent=2)}")
        
        if health_status["status"] == "healthy":
            logger.info("MLX Service initialized and ready")
        else:
            logger.error(f"Service initialization issues detected: {health_status}")
            
    except Exception as e:
        logger.error(f"Service initialization failed: {str(e)}")
        raise