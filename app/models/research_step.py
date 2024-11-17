from sqlalchemy import Column, Integer, String, DateTime, JSON, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from .base import Base
import json

class ResearchStep(Base):
    __tablename__ = 'research_steps'
    
    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey('projects.id'))
    step_number = Column(Integer)
    description = Column(String(1000))
    keywords = Column(JSON)
    methodology = Column(String(500))
    output_format = Column(String(200))
    status = Column(String(50))
    result = Column(JSON)
    executed_at = Column(DateTime)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    error_message = Column(String(500))
    progress_percentage = Column(Integer, default=0)
    
    project = relationship("Project", back_populates="steps")

    @property
    def keywords_list(self):
        if isinstance(self.keywords, str):
            return json.loads(self.keywords)
        return self.keywords or []

    @keywords_list.setter
    def keywords_list(self, value):
        if isinstance(value, list):
            self.keywords = json.dumps(value)
        else:
            self.keywords = value
