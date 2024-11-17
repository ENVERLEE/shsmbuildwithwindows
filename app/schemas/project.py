from pydantic import BaseModel
from typing import Optional, Dict, List
from datetime import datetime

class ProjectBase(BaseModel):
    title: str
    description: str
    evaluation_plan: str
    submission_format: str
    metadata_is: Optional[Dict] = {}

class ProjectCreate(BaseModel):
    title: str
    description: str
    evaluation_plan: Optional[str] = None
    submission_format: Optional[str] = None
    metadata1: Optional[Dict] = {}

class ProjectResponse(BaseModel):
    id: int
    title: str
    description: str
    created_at: datetime
    status: str

    class Config:
        from_attributes = True  

class Project(ProjectBase):
    id: int
    user_id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class ProjectWithSteps(Project):
    steps: List['ResearchStep']
