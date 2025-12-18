# API Contract: RAG Chatbot Backend

## Overview
This document defines the API contracts for the RAG chatbot backend service based on the functional requirements. The API follows REST principles with JSON request/response bodies.

## Base URL
```
https://rag-chatbot-backend.example.com/api/v1
```

## Common Headers
- `Content-Type: application/json`
- `Accept: application/json`
- `Authorization: Bearer {API_KEY}` (for backend-to-backend communication)

## Error Responses
All error responses follow this structure:
```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable error message",
    "details": "Optional detailed error information"
  }
}
```

## Endpoints

### 1. Global RAG Query
**Endpoint**: `POST /query`
**Description**: Submit a question about the book content for global RAG search

#### Request
```json
{
  "query": "What is the main principle of ROS 2 architecture?",
  "session_id": "optional-session-uuid",
  "include_citations": true
}
```

#### Request Validation
- `query`: Required, 1-2000 characters
- `session_id`: Optional, valid UUID format
- `include_citations`: Optional, boolean, defaults to true

#### Response (Success - 200 OK)
```json
{
  "response_id": "response-uuid",
  "query": "What is the main principle of ROS 2 architecture?",
  "answer": "The main principle of ROS 2 architecture is distributed computing with a DDS-based middleware layer that provides reliable message passing between nodes...",
  "citations": [
    {
      "section_title": "ROS 2 Architecture Overview",
      "section_path": "/module-1/ros2-introduction",
      "text_snippet": "ROS 2 uses DDS (Data Distribution Service) as its underlying middleware...",
      "similarity_score": 0.87,
      "page_number": 45
    }
  ],
  "confidence_score": 0.92,
  "processing_time_ms": 1250,
  "session_id": "session-uuid"
}
```

#### Response Validation
- `response_id`: Required, valid UUID
- `answer`: Required, non-empty string
- `citations`: Required array of citation objects
- `confidence_score`: Required, number between 0-1
- `processing_time_ms`: Required, positive number

#### Error Responses
- `400 Bad Request`: Invalid request format
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Processing error

### 2. Selected Text Query
**Endpoint**: `POST /query-selected`
**Description**: Submit a question about user-selected text only

#### Request
```json
{
  "query": "Explain this concept in more detail?",
  "selected_text": "The robot operating system (ROS) is a flexible framework for writing robot software...",
  "page_context": "/module-1/ros2-introduction",
  "session_id": "optional-session-uuid"
}
```

#### Request Validation
- `query`: Required, 1-2000 characters
- `selected_text`: Required, 1-5000 characters
- `page_context`: Required, string representing page location
- `session_id`: Optional, valid UUID format

#### Response (Success - 200 OK)
```json
{
  "response_id": "response-uuid",
  "query": "Explain this concept in more detail?",
  "answer": "The robot operating system (ROS) framework provides a collection of tools, libraries, and conventions that aim to simplify the task of creating complex and robust robot behavior...",
  "citations": [
    {
      "section_title": "ROS 2 Architecture Overview",
      "section_path": "/module-1/ros2-introduction",
      "text_snippet": "The robot operating system (ROS) is a flexible framework for writing robot software...",
      "similarity_score": 1.0,
      "page_number": null
    }
  ],
  "confidence_score": 0.85,
  "processing_time_ms": 980,
  "session_id": "session-uuid"
}
```

#### Response Validation
- Same as global query response

#### Error Responses
- `400 Bad Request`: Invalid request format
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Processing error

### 3. Health Check
**Endpoint**: `GET /health`
**Description**: Check the health status of the service

#### Response (Success - 200 OK)
```json
{
  "status": "healthy",
  "timestamp": "2025-12-18T10:30:00Z",
  "version": "1.0.0",
  "dependencies": {
    "qdrant": "connected",
    "postgres": "connected",
    "llm_provider": "available"
  }
}
```

#### Error Responses
- `503 Service Unavailable`: Service or dependencies unhealthy

### 4. Session Management
**Endpoint**: `POST /sessions`
**Description**: Create a new session for conversation context

#### Request
```json
{
  "user_id": "optional-user-uuid",
  "session_metadata": {
    "user_agent": "optional user agent string",
    "referrer": "optional referrer information"
  }
}
```

#### Response (Success - 201 Created)
```json
{
  "session_id": "new-session-uuid",
  "created_at": "2025-12-18T10:30:00Z",
  "expires_at": "2025-12-18T11:30:00Z"
}
```

## Security Considerations
1. All API keys must be passed via environment variables
2. CORS must be properly configured for browser requests
3. Rate limiting should be implemented to prevent abuse
4. Input validation must prevent injection attacks

## Performance Requirements
1. 90% of queries should respond within 3 seconds
2. API should handle concurrent requests efficiently
3. Response times should be included in all responses