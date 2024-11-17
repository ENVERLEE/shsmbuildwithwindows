from sqlalchemy import Column, Integer, String, DateTime, JSON, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from .base import Base

class Reference(Base):
    __tablename__ = 'reference_materials'
    
    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey('projects.id'))
    title = Column(String(200))
    authors = Column(JSON)
    publication_date = Column(DateTime)
    content = Column(String(2000))
    metadata1 = Column(JSON)
    embedding = Column(JSON)
    
    project = relationship("Project", back_populates="references")
