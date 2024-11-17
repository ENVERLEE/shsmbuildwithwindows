from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1 import router as api_v1_router
from app.core.config import settings
from app.core.logging_config import setup_logging
from app.core.middleware.error_handler import ErrorHandler, http_error_handler, generic_error_handler
from app.utils.database import create_tables
import logging

# 로깅 설정
setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 에러 핸들러 미들웨어 추가
app.middleware("http")(ErrorHandler())

# API 라우터 등록
app.include_router(api_v1_router, prefix=settings.API_V1_STR)

@app.on_event("startup")
def startup_event():
    logger.info("Starting up application...")
    create_tables()

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/")
async def root():
    return {
        "message": f"Welcome to {settings.PROJECT_NAME}",
        "version": settings.VERSION,
        "docs_url": "/docs"
    }

from app.core.env_validator import EnvValidator

@app.on_event("startup")
def validate_env():
    EnvValidator.validate_required_vars()
