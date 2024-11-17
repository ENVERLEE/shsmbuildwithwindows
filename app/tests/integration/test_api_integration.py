import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
import json
from datetime import datetime
from app.main import app
from app.models import User, Project
from app.utils.security import create_access_token

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def test_user(db: Session):
    user = User(
        email="test@example.com",
        hashed_password="hashed_password",
        full_name="Test User"
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

@pytest.fixture
def token(test_user):
    return create_access_token(data={"sub": str(test_user.id)})

def test_create_project(client, token):
    headers = {"Authorization": f"Bearer {token}"}
    data = {
        "title": "Test Project",
        "description": "Test Description",
        "evaluation_plan": "Test Plan",
        "submission_format": "Test Format",
        "metadata1": {}
    }
    response = client.post("/api/v1/projects/", json=data, headers=headers)
    assert response.status_code == 200
    assert response.json()["title"] == "Test Project"

def test_search_references(client, token):
    headers = {"Authorization": f"Bearer {token}"}
    params = {
        "project_id": 1,
        "keywords": ["AI", "Machine Learning"]
    }
    response = client.post("/api/v1/references/search", json=params, headers=headers)
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_get_reference_summary(client, token):
    headers = {"Authorization": f"Bearer {token}"}
    response = client.get("/api/v1/references/1/summary", headers=headers)
    if response.status_code == 404:
        assert response.json()["detail"] == "참고문헌을 찾을 수 없습니다."
    else:
        assert response.status_code == 200
        assert "title" in response.json()
        assert "summary" in response.json()
