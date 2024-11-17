import pytest
from unittest.mock import Mock, Mock, MagicMock
import sys
import os
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from app.services.research_service import ResearchService
from app.core.errors.exceptions import APIError

class AsyncSessionMock(MagicMock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.begin = Mock()
        self.commit = Mock()
        self.rollback = Mock()
        self.flush = Mock()
        self.add = Mock()

    def __aenter__(self):
        return self

    def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

@pytest.fixture
def mock_services():
    class MockAsyncContextManager:
        def __aenter__(self):
            return self
            
        def __aexit__(self, exc_type, exc_val, exc_tb):
            pass
            
    # 비동기 컨텍스트 매니저 생성
    ctx_manager = MockAsyncContextManager()
    
    # DB 모킹
    db_mock = Mock()
    db_mock.begin = Mock(return_value=ctx_manager)
    db_mock.flush = Mock()
    db_mock.add = Mock()
    
    # 기타 서비스 모킹
    mlx_mock = Mock()
    perplexity_mock = Mock()
    
    return db_mock, mlx_mock, perplexity_mock

def test_create_project_success(mock_services):
    db_mock, mlx_mock, perplexity_mock = mock_services
    service = ResearchService(db_mock, mlx_mock, perplexity_mock)
    
    # Mock response
    mock_response = Mock()
    mock_response.content = "## 1단계\n**설명**: 테스트\n**키워드**: test\n**방법론**: test method"
    service.anthropic = Mock()
    service.anthropic.messages.create.return_value = mock_response
    
    project_data = {
        "title": "Test Project",
        "description": "Test Description",
        "evaluation_plan": "Test Plan",
        "submission_format": "Test Format",
        "metadata1": {},
        "research_field": "안보"
    }
    
    result = service.create_project(1, project_data)
    assert result.title == "Test Project"
    assert db_mock.add.called


def test_create_project_api_error(mock_services):
    db_mock, mlx_mock, perplexity_mock = mock_services
    service = ResearchService(db_mock, mlx_mock, perplexity_mock)
    
    def mock_transaction():
        return db_mock
    db_mock.begin.return_value = mock_transaction()
    
    project_data = {
        "title": "Test Project",
        "description": "Test Description",
        "evaluation_plan": "Test Plan",
        "submission_format": "Test Format",
        "metadata1": {},
        "research_field": "안보"
    }
    
    service.anthropic = Mock()
    service.anthropic.messages.create.side_effect = Exception("API Error")
    
    with pytest.raises(HTTPException) as exc_info:
        service.create_project(1, project_data)
    assert exc_info.value.status_code == 500