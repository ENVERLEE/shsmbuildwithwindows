import logging
import sys
from pathlib import Path
from loguru import logger

class InterceptHandler(logging.Handler):
    def emit(self, record):
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

def setup_logging():
    logging.basicConfig(handlers=[InterceptHandler()], level=0)
    for logger_name in logging.root.manager.loggerDict:
        if logger_name.startswith("uvicorn"):
            logging_logger = logging.getLogger(logger_name)
            logging_logger.handlers = []

    logger.configure(
        handlers=[
            {"sink": sys.stdout, "format": "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"},
            {"sink": "logs/app.log", "rotation": "500 MB", "retention": "1 week"},
        ]
    )
