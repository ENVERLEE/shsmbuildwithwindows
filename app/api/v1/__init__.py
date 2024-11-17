from fastapi import APIRouter
from . import auth, projects, references

router = APIRouter()

router.include_router(auth.router, prefix="/auth", tags=["Authentication"])
router.include_router(projects.router, prefix="/projects", tags=["Projects"])
router.include_router(references.router, prefix="/references", tags=["References"])
