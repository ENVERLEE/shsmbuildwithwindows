from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from typing import Union
import logging

logger = logging.getLogger(__name__)

class ErrorHandler:
    async def __call__(self, request: Request, call_next):
        try:
            return await call_next(request)
        except HTTPException as e:
            return JSONResponse(
                status_code=e.status_code,
                content={"detail": e.detail}
            )
        except Exception as e:
            logger.exception("Unexpected error occurred")
            return JSONResponse(
                status_code=500,
                content={"detail": "Internal server error"}
            )

async def http_error_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

async def generic_error_handler(request: Request, exc: Exception):
    logger.exception("Unexpected error occurred")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )
