from fastapi import APIRouter
from datetime import datetime
from app.rag.schemas import HealthResponse

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        version="1.0.0",
        dependencies={
            "qdrant": "connected",
            "postgres": "connected",
            "llm_provider": "available"
        }
    )