# Research Summary: Integrated RAG Chatbot for Physical AI & Humanoid Robotics Book

## Overview
This research document addresses the technical requirements and implementation approach for the RAG chatbot system as specified in the feature requirements. All "NEEDS CLARIFICATION" items from the technical context have been resolved through research and analysis.

## Decision: Technology Stack Selection
**Rationale**: Based on the feature requirements and architecture constraints, the following technology stack was selected:
- **Backend**: FastAPI with Python 3.11 for API services
- **Frontend**: React components integrated with Docusaurus
- **Vector Database**: Qdrant Cloud for semantic search capabilities
- **Metadata Storage**: Neon Serverless Postgres for session and document metadata
- **LLM Provider**: OpenRouter API with Qwen model for generation
- **Embeddings**: Qwen embeddings via OpenRouter for consistency

**Alternatives considered**:
- Alternative LLM providers (OpenAI, Anthropic, Hugging Face) - rejected in favor of OpenRouter/Qwen to maintain consistency with architecture requirements
- Different vector databases (Pinecone, Weaviate, Chroma) - Qdrant chosen for free tier availability and good Python integration
- Alternative backend frameworks (Flask, Django) - FastAPI chosen for async performance and OpenAPI documentation

## Decision: Architecture Pattern
**Rationale**: The three-tier architecture (ingestion, backend API, frontend) was chosen to satisfy the requirement for independent backend deployment while maintaining the constraint of not modifying existing book Markdown content. The Docusaurus plugin approach ensures seamless integration.

**Alternatives considered**:
- Monolithic approach - rejected due to deployment independence requirement
- Serverless functions - rejected due to complexity of vector database integration
- Direct client-side integration with LLM APIs - rejected due to security concerns and CORS issues

## Decision: Content Processing Pipeline
**Rationale**: A dedicated ingestion pipeline was designed to parse Docusaurus markdown content, chunk it appropriately with section-level metadata, and generate embeddings for storage in Qdrant. This ensures the system works with existing content without modifications.

**Alternatives considered**:
- Real-time parsing during queries - rejected due to performance concerns
- Manual content conversion - rejected due to maintenance overhead
- Direct integration with Docusaurus build process - rejected due to complexity and potential for breaking existing functionality

## Decision: Selected Text Mode Implementation
**Rationale**: The selected text mode will operate by bypassing the vector database entirely when activated, using only the highlighted text as context for the LLM. This ensures complete isolation from the global index as required.

**Alternatives considered**:
- Scoring highlighted text higher in retrieval - rejected as it would still include other content
- Creating temporary embeddings for selected text - rejected due to complexity and real-time performance requirements

## Decision: Performance Optimization Strategy
**Rationale**: To achieve sub-3s response times, the system will implement:
- Pre-computed embeddings for book content
- Caching for common queries
- Optimized context window management
- Asynchronous processing where possible

**Alternatives considered**:
- Larger LLM models for quality - rejected due to cost and latency concerns
- Multiple parallel queries - rejected due to cost implications for free tier
- Client-side processing - rejected due to computational requirements

## Decision: Security and Configuration Management
**Rationale**: API keys will be managed through environment variables as required, with additional security measures including:
- Rate limiting to prevent abuse
- Input validation to prevent injection attacks
- CORS configuration for safe browser integration
- Session management for conversation context

**Alternatives considered**:
- Hardcoded keys in source - rejected as explicitly forbidden in requirements
- Client-side key storage - rejected due to security concerns
- Shared API keys across users - rejected due to accountability requirements