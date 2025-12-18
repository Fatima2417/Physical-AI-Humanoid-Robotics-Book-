from fastapi import APIRouter
from . import query, health

router = APIRouter()
router.include_router(query.router, tags=["query"])
router.include_router(health.router, tags=["health"])