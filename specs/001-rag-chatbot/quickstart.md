# Quickstart Guide: RAG Chatbot for Physical AI & Humanoid Robotics Book

## Overview
This guide provides step-by-step instructions to set up, configure, and run the RAG chatbot system for the Physical AI & Humanoid Robotics book.

## Prerequisites
- Python 3.11+ with pip
- Node.js 18+ with npm
- Access to OpenRouter API (for Qwen model and embeddings)
- Qdrant Cloud account (free tier available)
- Neon Serverless Postgres account (free tier available)

## Environment Setup

### 1. Clone the Repository
```bash
git clone <repository-url>
cd <repository-directory>
git checkout 001-rag-chatbot
```

### 2. Set Up Backend Environment
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Set Up Environment Variables
Create a `.env` file in the backend directory with the following variables:
```env
OPENROUTER_API_KEY=your_openrouter_api_key
QDRANT_URL=your_qdrant_cloud_url
QDRANT_API_KEY=your_qdrant_api_key
NEON_DATABASE_URL=your_neon_postgres_connection_string
DEBUG=false
LOG_LEVEL=INFO
MAX_QUERY_LENGTH=2000
MAX_SELECTED_TEXT_LENGTH=5000
RESPONSE_TIMEOUT=30
```

## Backend Setup

### 1. Install Backend Dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 2. Run Database Migrations (if applicable)
```bash
# Follow any database migration instructions specific to your setup
```

### 3. Start Backend Service
```bash
cd backend
python -m src.api.main
```
The backend will start on `http://localhost:8000`

## Content Ingestion

### 1. Prepare Book Content
Ensure your Docusaurus book content is available in the `Book/docs` directory (as per project constitution).

### 2. Install Ingestion Dependencies
```bash
cd ingestion
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt  # if ingestion has its own requirements
```

### 3. Run Content Ingestion
```bash
cd ingestion
python -m scripts.ingest_content --source-path ../Book/docs --target-qdrant --chunk-size 512
```

This process will:
- Parse all markdown files in the book
- Chunk content while preserving section metadata
- Generate embeddings using Qwen via OpenRouter
- Store chunks in Qdrant vector database
- Store document metadata in Neon Postgres

## Frontend Integration

### 1. Install Frontend Dependencies
```bash
cd frontend
npm install
```

### 2. Build the Docusaurus Plugin
```bash
cd Book  # The main Docusaurus book directory
npm install ../frontend/docusaurus-plugin  # Adjust path as needed
```

### 3. Configure Docusaurus
Add the plugin to your `docusaurus.config.js`:
```js
module.exports = {
  // ... other config
  plugins: [
    // ... other plugins
    [
      '../frontend/docusaurus-plugin',  // Adjust path as needed
      {
        backendUrl: process.env.BACKEND_URL || 'http://localhost:8000',
        // Additional plugin configuration
      }
    ]
  ],
  // ... rest of config
};
```

### 4. Start Docusaurus Development Server
```bash
cd Book
npm run start
```

## Testing the System

### 1. Verify Backend Health
```bash
curl http://localhost:8000/api/v1/health
```

### 2. Test a Query
```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is ROS 2?",
    "include_citations": true
  }'
```

### 3. Test Selected Text Mode
```bash
curl -X POST http://localhost:8000/api/v1/query-selected \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Explain this concept?",
    "selected_text": "ROS 2 is a flexible framework for writing robot software",
    "page_context": "/module-1/ros2-introduction"
  }'
```

## Configuration Options

### Backend Configuration
- `BACKEND_HOST`: Host for the backend service (default: 0.0.0.0)
- `BACKEND_PORT`: Port for the backend service (default: 8000)
- `MAX_CONCURRENT_REQUESTS`: Maximum concurrent requests (default: 10)
- `CACHE_SIZE`: Size of the query response cache (default: 1000)

### Frontend Configuration
- `CHATBOT_POSITION`: Position of the chatbot widget (default: "bottom-right")
- `THEME_COLOR`: Primary color for the chatbot UI (default: "#2563eb")
- `ENABLE_SESSIONS`: Whether to enable conversation sessions (default: true)

## Deployment

### Backend Deployment
The backend can be deployed to platforms like:
- Railway: `npx railway deploy`
- Fly.io: `flyctl deploy`
- Render: Set up via Render dashboard

### Frontend Integration
The frontend is integrated into the existing Docusaurus build, so it deploys with the book.

## Troubleshooting

### Common Issues

1. **Embedding Generation Fails**
   - Check OpenRouter API key validity
   - Verify rate limits are not exceeded
   - Confirm Qdrant connection details

2. **Slow Response Times**
   - Check network connectivity to external services
   - Verify Qdrant performance on free tier
   - Review query complexity

3. **CORS Issues**
   - Ensure backend CORS configuration allows your frontend origin
   - Check that API endpoints are properly configured

### Useful Commands
```bash
# Check backend logs
cd backend && python -m src.api.main --log-level DEBUG

# Verify content ingestion
cd ingestion && python -c "from src.loaders.vector_loader import check_connection; check_connection()"

# Test API endpoints
curl -X GET http://localhost:8000/api/v1/health
```

## Next Steps
1. Customize the chatbot UI to match your book's styling
2. Fine-tune the RAG parameters for optimal performance
3. Add analytics to track usage and effectiveness
4. Implement additional features based on user feedback