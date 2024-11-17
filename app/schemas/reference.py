from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime

class ReferenceBase(BaseModel):
    title: str
    authors: List[str]
    content: str
    publication_date: datetime
    metadata1: Optional[Dict] = {}

class ReferenceCreate(ReferenceBase):
    pass

class Reference(ReferenceBase):
    id: int
    project_id: int
    embedding: Optional[List[float]] = None

    class Config:
        from_attributes = True
