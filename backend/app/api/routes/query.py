from fastapi import APIRouter, HTTPException
from typing import Optional
import time
from app.rag.schemas import QueryRequest, QuerySelectedRequest, QueryResponse
from app.rag.rag_service import rag_service

router = APIRouter()

@router.post("/query", response_model=QueryResponse)
async def global_query(request: QueryRequest):
    """Handle global RAG query - search across entire book corpus"""
    try:
        start_time = time.time()

        # Process the query using RAG service
        result = await rag_service.process_global_query(
            query_text=request.query,
            session_id=request.session_id,
            include_citations=request.include_citations
        )

        processing_time = int((time.time() - start_time) * 1000)

        return QueryResponse(
            response_id=result["response_id"],
            query=request.query,
            answer=result["answer"],
            citations=result["citations"],
            confidence_score=result["confidence_score"],
            processing_time_ms=processing_time,
            session_id=request.session_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/query-selected", response_model=QueryResponse)
async def selected_text_query(request: QuerySelectedRequest):
    """Handle selected text query - answer based only on provided text"""
    try:
        start_time = time.time()

        # Process the query using only selected text
        result = await rag_service.process_selected_text_query(
            query_text=request.query,
            selected_text=request.selected_text,
            page_context=request.page_context,
            session_id=request.session_id
        )

        processing_time = int((time.time() - start_time) * 1000)

        return QueryResponse(
            response_id=result["response_id"],
            query=request.query,
            answer=result["answer"],
            citations=result["citations"],
            confidence_score=result["confidence_score"],
            processing_time_ms=processing_time,
            session_id=request.session_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))