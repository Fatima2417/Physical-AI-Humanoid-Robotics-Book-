# Implementation Plan: Integrated RAG Chatbot for Physical AI & Humanoid Robotics Book

**Branch**: `001-rag-chatbot` | **Date**: 2025-12-18 | **Spec**: [link to spec](./spec.md)
**Input**: Feature specification from `/specs/001-rag-chatbot/spec.md`

**Note**: This template is filled in by the `/sp.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Implementation of a Retrieval-Augmented Generation (RAG) chatbot that enables users to ask questions about the Physical AI & Humanoid Robotics book content. The system includes semantic search over the entire book corpus, selected text mode for context-limited queries, and citation references in responses. The architecture separates frontend (Docusaurus React component) from backend (FastAPI service) with vector storage in Qdrant and metadata in Neon Postgres.

## Technical Context

**Language/Version**: Python 3.11 (backend), JavaScript/TypeScript (frontend), Node.js 18+
**Primary Dependencies**: FastAPI, OpenAI SDK, Qdrant client, Neon Postgres driver, React, Docusaurus
**Storage**: Qdrant Cloud (vector database), Neon Serverless Postgres (metadata), OpenRouter API (LLM)
**Testing**: pytest (backend), Jest/React Testing Library (frontend), integration tests
**Target Platform**: Web application (frontend: browser, backend: cloud deployment)
**Project Type**: Web application (frontend + backend separation)
**Performance Goals**: <3 second response time for 90% of queries, handle concurrent users
**Constraints**: Free-tier resource usage, no modification of existing book Markdown, <200ms UI response
**Scale/Scope**: Single book corpus, multiple concurrent users, session-based conversations

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

1. **Technical Accuracy and Zero Hallucinations**: The RAG system must ensure responses are grounded only in book content with zero hallucinations. This aligns with the constitution's requirement for technical accuracy.
2. **Reproducibility and Practical Implementation**: All components must be deployable and testable using standard cloud services (Qdrant, Neon, OpenRouter). Implementation must be reproducible with documented environment setup.
3. **Docusaurus-First Documentation**: The chatbot component must integrate seamlessly with the existing Docusaurus structure without modifying Markdown content, aligning with the constitution's requirement.
4. **Complete and Production-Ready Content**: The RAG system must be fully functional with proper error handling, monitoring, and failover mechanisms.

## Project Structure

### Documentation (this feature)

```text
specs/001-rag-chatbot/
├── plan.md              # This file (/sp.plan command output)
├── research.md          # Phase 0 output (/sp.plan command)
├── data-model.md        # Phase 1 output (/sp.plan command)
├── quickstart.md        # Phase 1 output (/sp.plan command)
├── contracts/           # Phase 1 output (/sp.plan command)
└── tasks.md             # Phase 2 output (/sp.tasks command - NOT created by /sp.plan)
```

### Source Code (repository root)

```text
backend/
├── src/
│   ├── models/
│   │   ├── query.py          # Query data models
│   │   ├── response.py       # Response data models
│   │   └── document.py       # Document/chunk models
│   ├── services/
│   │   ├── rag_service.py    # Core RAG logic
│   │   ├── embedding_service.py # Embedding generation
│   │   ├── retrieval_service.py # Vector search
│   │   └── llm_service.py    # LLM interaction
│   ├── api/
│   │   ├── main.py           # FastAPI app
│   │   ├── routes/
│   │   │   ├── query.py      # Query endpoints
│   │   │   └── health.py     # Health check
│   │   └── middleware/
│   │       └── cors.py       # CORS configuration
│   └── config/
│       ├── settings.py       # Configuration management
│       └── database.py       # Database connections
├── tests/
│   ├── unit/
│   ├── integration/
│   └── contract/
└── requirements.txt

frontend/
├── src/
│   ├── components/
│   │   ├── Chatbot.jsx       # Main chatbot component
│   │   ├── ChatWindow.jsx    # Chat interface
│   │   ├── Message.jsx       # Individual message
│   │   └── Citation.jsx      # Citation display
│   ├── hooks/
│   │   └── useChatbot.js     # Chatbot state management
│   ├── services/
│   │   └── api.js            # API client
│   └── styles/
│       └── chatbot.css       # Chatbot styling
└── docusaurus-plugin/
    └── index.js              # Docusaurus plugin entry

ingestion/
├── src/
│   ├── parsers/
│   │   └── markdown_parser.py # Parse Docusaurus markdown
│   ├── chunkers/
│   │   └── document_chunker.py # Content chunking logic
│   ├── processors/
│   │   └── embedding_generator.py # Generate embeddings
│   └── loaders/
│       └── vector_loader.py  # Load to Qdrant
└── scripts/
    └── ingest_content.py     # Main ingestion script
```

**Structure Decision**: The system is structured as a web application with clear separation between frontend and backend. The frontend integrates as a Docusaurus plugin to avoid modifying existing Markdown content. The backend provides API services with separate ingestion pipeline for processing book content into vector database.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| Multiple service architecture | Required for separation of concerns and independent deployment | Single service would violate requirement for independent backend deployment |
| External vector database | Required for semantic search capabilities | In-memory search would not scale to full book corpus |
