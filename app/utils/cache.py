from functools import lru_cache, wraps
from app.core.config.settings import settings
import time
from typing import Any, Callable

def timed_lru_cache(seconds: int, maxsize: int = None):
    def wrapper_factory(func):
        func = lru_cache(maxsize=maxsize)(func)
        func.lifetime = seconds
        func.expiration = time.time() + seconds

        @wraps(func)
        def wrapped_func(*args, **kwargs):
            if time.time() >= func.expiration:
                func.cache_clear()
                func.expiration = time.time() + func.lifetime
            return func(*args, **kwargs)

        return wrapped_func
    return wrapper_factory

class CacheManager:
    @timed_lru_cache(seconds=settings.CACHE_TTL, maxsize=settings.CACHE_MAX_SIZE)
    async def get_project_details(self, project_id: int) -> dict:
        # 프로젝트 상세 정보를 캐시
        pass

    @timed_lru_cache(seconds=settings.CACHE_TTL, maxsize=settings.CACHE_MAX_SIZE)
    async def get_reference_details(self, reference_id: int) -> dict:
        # 참고문헌 상세 정보를 캐시
        pass
