from pydantic_settings import BaseSettings
from typing import Optional

class AppSettings(BaseSettings):
    # API 설정
    API_BASE_URL: str = "http://localhost:8000/api/v1"
    API_VERSION: str = "v1"
    
    # 타임아웃 설정
    TIMEOUT_RESEARCH_PLAN: int = 300
    TIMEOUT_REFERENCE_SEARCH: int = 120
    TIMEOUT_CONTENT_ANALYSIS: int = 180
    
    # 데이터베이스 설정
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_NAME: str
    DB_USER: str
    DB_PASSWORD: str
    
    # 캐시 설정
    CACHE_TTL: int = 3600
    CACHE_MAX_SIZE: int = 100
    
    # API 키
    COHERE_API_KEY: str
    PERPLEXITY_API_KEY: str
    XAI_API_KEY: str
    
    class Config:
        env_file = ".env"

settings = AppSettings()
