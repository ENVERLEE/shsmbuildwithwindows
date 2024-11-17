from sqlalchemy import Column, Integer, String, DateTime, JSON, ForeignKey, Numeric
from sqlalchemy.orm import relationship
from datetime import datetime
from .base import Base

class Project(Base):
    __tablename__ = 'projects'
    
    id = Column(Integer, primary_key=True)
    title = Column(String(200), nullable=False)
    description = Column(String(1000))
    evaluation_plan = Column(String(1000))
    submission_format = Column(String(500))
    metadata1 = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    user_id = Column(Integer, ForeignKey('users.id'))
    
    # 새로운 필드 추가
    research_field = Column(String(100), default='안보')
    evaluation_status = Column(String(50), default='pending')
    final_score = Column(Numeric(5, 2))
    submission_date = Column(DateTime)
    total_steps = Column(Integer, default=0)
    completed_steps = Column(Integer, default=0)
    
    steps = relationship("ResearchStep", back_populates="project")
    references = relationship("Reference", back_populates="project")
    user = relationship("User", back_populates="projects")