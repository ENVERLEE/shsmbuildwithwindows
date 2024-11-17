from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Dict, Optional
from app.models import Reference
from app.services import ReferenceService, PerplexityService, MLXService
from app.utils.database import get_db
from app.utils.security import get_current_user
from pydantic import BaseModel
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()

class SearchRequest(BaseModel):
    project_id: int
    keywords: List[str]

@router.post("/search")
def search_references(
    search_request: SearchRequest,
    db: Session = Depends(get_db),
    current_user: int = Depends(get_current_user)
):
    perplexity_service = PerplexityService(os.getenv("PERPLEXITY_API_KEY"))
    cohere_service = MLXService()
    reference_service = ReferenceService(db, perplexity_service, cohere_service)
    
    try:
        references = reference_service.search_and_save_references(
            search_request.project_id, 
            search_request.keywords
        )
        return [
            {
                "id": ref.id,
                "title": ref.title,
                "authors": ref.authors,
                "publication_date": ref.publication_date,
                "content": ref.content,
                "metadata": ref.metadata
            }
            for ref in references
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
