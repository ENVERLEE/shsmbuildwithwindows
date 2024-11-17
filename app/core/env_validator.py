from typing import List
import os
from fastapi import HTTPException

class EnvValidator:
    @staticmethod
    def validate_required_vars():
        required_vars = [
            'DATABASE_URL',
            'SECRET_KEY',
            'ALGORITHM',
            'ACCESS_TOKEN_EXPIRE_MINUTES',
            'COHERE_API_KEY',
            'PERPLEXITY_API_KEY',
            'ANTHROPIC_API_KEY',
            'VOYAGE_API_KEY',
            'XAI_API_KEY'
        ]
        
        missing_vars = []
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            raise HTTPException(
                status_code=500,
                detail=f"Missing required environment variables: {', '.join(missing_vars)}"
            )
