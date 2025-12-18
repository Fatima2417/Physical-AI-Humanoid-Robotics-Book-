# Data Model: Integrated RAG Chatbot for Physical AI & Humanoid Robotics Book

## Overview
This document defines the data models for the RAG chatbot system, including entities, their attributes, relationships, and validation rules based on the feature requirements.

## Core Entities

### Query
**Description**: Represents a user's question about the book content

**Attributes**:
- `id`: String (UUID) - Unique identifier for the query
- `content`: String (required, 1-2000 characters) - The user's question text
- `mode`: String (enum: "global", "selected_text") - Query mode (global RAG vs selected text)
- `selected_text`: String (optional, 0-5000 characters) - Text selected by user for selected_text mode
- `page_context`: String (optional, 0-500 characters) - Page URL or identifier where query originated
- `timestamp`: DateTime - When the query was submitted
- `session_id`: String (optional, UUID) - Session identifier for conversation context

**Validation Rules**:
- Content must not be empty when mode is "global"
- Selected_text must be provided when mode is "selected_text"
- Content length must be between 1-2000 characters

### Response
**Description**: The system's answer to a user's query

**Attributes**:
- `id`: String (UUID) - Unique identifier for the response
- `query_id`: String (UUID, required) - Reference to the associated query
- `content`: String (required) - The generated response text
- `citations`: Array of Citation objects - References to source material
- `confidence_score`: Number (0-1) - Confidence level in the response accuracy
- `processing_time_ms`: Number - Time taken to generate the response
- `timestamp`: DateTime - When the response was generated

**Validation Rules**:
- Content must be grounded in retrieved context (no hallucinations)
- Citations must reference actual book content
- Confidence score must be between 0 and 1

### Citation
**Description**: Reference to source material used in a response

**Attributes**:
- `id`: String (UUID) - Unique identifier for the citation
- `response_id`: String (UUID, required) - Reference to the associated response
- `section_title`: String (required) - Title of the referenced section
- `section_path`: String (required) - Path to the document/section in the book
- `text_snippet`: String (required) - The actual text that was referenced
- `similarity_score`: Number (0-1) - How relevant this citation was to the query
- `page_number`: Number (optional) - Page number if applicable

**Validation Rules**:
- Section path must correspond to an actual document in the corpus
- Text snippet must be an exact or close match to content in the book

### DocumentChunk
**Description**: A chunk of book content stored in the vector database

**Attributes**:
- `id`: String (UUID) - Unique identifier for the chunk
- `document_id`: String (required) - Identifier for the source document
- `content`: String (required) - The text content of this chunk
- `section_title`: String (required) - Title of the section this chunk belongs to
- `section_path`: String (required) - Path to the document in the book structure
- `page_number`: Number (optional) - Page number if applicable
- `chunk_order`: Number - Order of this chunk within the document
- `embedding`: Array of Numbers - Vector embedding of the content
- `metadata`: Object - Additional metadata for retrieval

**Validation Rules**:
- Content length should be optimized for context window (typically 500-1000 tokens)
- Embedding must be properly formatted vector representation

### Session
**Description**: Optional conversation context for maintaining dialogue state

**Attributes**:
- `id`: String (UUID) - Unique identifier for the session
- `user_id`: String (optional, UUID) - Identifier for the user (if available)
- `created_at`: DateTime - When the session was created
- `last_activity`: DateTime - When the session was last used
- `query_history`: Array of Query IDs - History of queries in this session
- `active`: Boolean - Whether the session is still active

**Validation Rules**:
- Sessions can be anonymous (no user_id)
- Session should expire after period of inactivity

### DocumentMetadata
**Description**: Metadata about documents in the book corpus

**Attributes**:
- `id`: String (UUID) - Unique identifier for the metadata record
- `document_path`: String (required) - Path to the document in the book structure
- `title`: String (required) - Title of the document
- `section_hierarchy`: Array of Strings - Hierarchy of sections (e.g., ["Chapter 1", "Section 1.2"])
- `word_count`: Number - Total word count of the document
- `token_count`: Number - Total token count of the document
- `last_modified`: DateTime - When the source document was last updated
- `embedding_status`: String (enum: "pending", "processing", "completed", "failed") - Status of embedding generation

**Validation Rules**:
- Document path must correspond to an actual file in the book
- All metadata should be accurate and up-to-date

## Relationships

```
Query (1) -- (1) Session
Query (1) -- (1) Response
Response (1) -- (*) Citation
Query (*) -- (*) DocumentChunk (via retrieval)
DocumentChunk (1) -- (1) DocumentMetadata
```

## State Transitions

### DocumentChunk States
- `pending` → `processing` → `completed` | `failed`
- Embedding generation workflow

### Session States
- `active` → `inactive` (after inactivity timeout)
- Session lifecycle management

## Constraints

1. **Integrity Constraints**:
   - All citations must reference valid document chunks
   - Query-response relationships must be maintained
   - Document paths must exist in the book corpus

2. **Performance Constraints**:
   - Document chunks should be sized appropriately for LLM context windows
   - Embeddings must be efficiently searchable

3. **Quality Constraints**:
   - Response content must be grounded in source material
   - Citations must accurately reference source content
   - No hallucinations allowed in responses