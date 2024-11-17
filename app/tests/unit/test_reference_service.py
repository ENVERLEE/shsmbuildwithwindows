import pytest
from datetime import datetime
from unittest.mock import Mock, AsyncMock
from sqlalchemy.orm import Session
from app.models import Reference
from app.services.reference_service import ReferenceService
from app.services.perplexity_service import PerplexityService
from app.services.cohere_service import CohereService

@pytest.fixture
def db_session():
    return Mock(spec=Session)

@pytest.fixture
def perplexity_service():
    service = Mock(spec=PerplexityService)
    service.search_references = AsyncMock()
    return service

@pytest.fixture
def cohere_service():
    service = Mock(spec=CohereService)
    service.generate_embeddings = AsyncMock()
    service.analyze_content = AsyncMock()
    return service

@pytest.fixture
def reference_service(db_session, perplexity_service, cohere_service):
    return ReferenceService(db_session, perplexity_service, cohere_service)

@pytest.mark.asyncio
async def test_search_and_save_references(reference_service, perplexity_service, cohere_service):
    project_id = 1
    keywords = ["AI", "Machine Learning"]
    mock_references = [
        {
            "title": "Test Paper",
            "authors": ["Author 1", "Author 2"],
            "abstract": "Test abstract",
            "publication_date": "2024-01-01T00:00:00",
            "journal": "Test Journal",
            "doi": "10.1234/test",
            "url": "http://test.com",
            "citations": 10
        }
    ]
    mock_embedding = [[0.1, 0.2, 0.3]]
    
    perplexity_service.search_references.return_value = mock_references
    cohere_service.generate_embeddings.return_value = mock_embedding
    
    result = await reference_service.search_and_save_references(project_id, keywords)
    
    assert len(result) == 1
    reference_service.db.add.assert_called_once()
    reference_service.db.commit.assert_called_once()
    
    saved_reference = reference_service.db.add.call_args[0][0]
    assert isinstance(saved_reference, Reference)
    assert saved_reference.project_id == project_id
    assert saved_reference.title == mock_references[0]["title"]
    assert saved_reference.embedding == mock_embedding[0]
