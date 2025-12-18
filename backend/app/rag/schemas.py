from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime
from uuid import UUID, uuid4
import re

class Query(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    content: str = Field(..., min_length=1, max_length=2000)
    mode: str = Field(default="global", pattern=r"^(global|selected_text)$")
    selected_text: Optional[str] = Field(default="", max_length=5000)
    page_context: Optional[str] = Field(default="", max_length=500)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    session_id: Optional[str] = Field(default="", pattern=r"^[a-f0-9]{8}-[a-f0-9]{4}-4[a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}$|^$")

    def model_post_init(self, __context):
        # Validate content based on mode
        if self.mode == "global" and not self.content.strip():
            raise ValueError("Content must not be empty when mode is 'global'")
        if self.mode == "selected_text" and not self.selected_text.strip():
            raise ValueError("Selected_text must be provided when mode is 'selected_text'")

class Citation(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    response_id: str
    section_title: str
    section_path: str
    text_snippet: str
    similarity_score: float = Field(ge=0.0, le=1.0)
    page_number: Optional[int] = None

class Response(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    query_id: str
    content: str
    citations: List[Citation] = []
    confidence_score: float = Field(ge=0.0, le=1.0)
    processing_time_ms: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class DocumentChunk(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    document_id: str
    content: str
    section_title: str
    section_path: str
    page_number: Optional[int] = None
    chunk_order: int
    embedding: Optional[List[float]] = None
    metadata: Dict = {}

class Session(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: Optional[str] = Field(default="", pattern=r"^[a-f0-9]{8}-[a-f0-9]{4}-4[a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}$|^$")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_activity: datetime = Field(default_factory=datetime.utcnow)
    query_history: List[str] = []
    active: bool = True

class DocumentMetadata(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    document_path: str
    title: str
    section_hierarchy: List[str] = []
    word_count: int = 0
    token_count: int = 0
    last_modified: datetime = Field(default_factory=datetime.utcnow)
    embedding_status: str = Field(default="pending", pattern=r"^(pending|processing|completed|failed)$")

# Request/Response schemas for API endpoints
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    session_id: Optional[str] = Field(default="", pattern=r"^[a-f0-9]{8}-[a-f0-9]{4}-4[a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}$|^$")
    include_citations: bool = True

class QuerySelectedRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    selected_text: str = Field(..., min_length=1, max_length=5000)
    page_context: str = Field(default="", max_length=500)
    session_id: Optional[str] = Field(default="", pattern=r"^[a-f0-9]{8}-[a-f0-9]{4}-4[a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}$|^$")

class QueryResponse(BaseModel):
    response_id: str
    query: str
    answer: str
    citations: List[Citation]
    confidence_score: float
    processing_time_ms: int
    session_id: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    dependencies: Dict[str, str]

class SessionRequest(BaseModel):
    user_id: Optional[str] = Field(default="", pattern=r"^[a-f0-9]{8}-[a-f0-9]{4}-4[a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}$|^$")
    session_metadata: Optional[Dict] = {}

class SessionResponse(BaseModel):
    session_id: str
    created_at: str
    expires_at: str