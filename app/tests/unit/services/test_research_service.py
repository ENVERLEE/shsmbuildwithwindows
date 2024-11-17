# tests/unit/services/test_research_service.py

import pytest
from unittest.mock import Mock, AsyncMock, patch
from test.app.services.research_service import ResearchService
from test.app.core.errors.exceptions import ResourceNotFoundError, APIError
from test.app.models.project import Project

@pytest.fixture
def mock_services():
    class MockAsyncContextManager:
        async def __aenter__(self):
            return self
            
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass
            
    # 비동기 컨텍스트 매니저 생성
    ctx_manager = MockAsyncContextManager()
    
    # DB 모킹
    db_mock = AsyncMock()
    db_mock.begin = AsyncMock(return_value=ctx_manager)
    db_mock.flush = AsyncMock()
    db_mock.add = Mock()
    
    # 기타 서비스 모킹
    mlx_mock = AsyncMock()
    perplexity_mock = AsyncMock()
    
    return db_mock, mlx_mock, perplexity_mock

@pytest.mark.asyncio
async def test_create_project_success(mock_services):
    db_mock, mlx_mock, perplexity_mock = mock_services
    service = ResearchService(db_mock, mlx_mock, perplexity_mock)
    
    # Mock response
    mock_response = AsyncMock()
    mock_response.content = "## 1단계\n**설명**: 테스트\n**키워드**: test\n**방법론**: test method"
    service.anthropic = AsyncMock()
    service.anthropic.messages.create.return_value = mock_response
    
    project_data = {
        "title": "Test Project",
        "description": "Test Description",
        "evaluation_plan": "Test Plan",
        "submission_format": "Test Format",
        "metadata1": {},
        "research_field": "안보"
    }
    
    result = await service.create_project(1, project_data)
    assert result.title == "Test Project"
    assert db_mock.add.called

@pytest.mark.asyncio
async def test_create_project_api_error(mock_services):
    db_mock, anthropic_mock, perplexity_mock = mock_services
    service = ResearchService(db_mock, anthropic_mock, perplexity_mock)
    
    project_data = {
        "title": "Test Project",
        "description": "Test Description",
        "evaluation_plan": "Test Plan",
        "submission_format": "Test Format",
        "metadata1": {},
        "research_field": "안보"
    }
    
    anthropic_mock.messages.create.side_effect = Exception("API Error")
    
    with pytest.raises(APIError):
        await service.create_project(1, project_data)