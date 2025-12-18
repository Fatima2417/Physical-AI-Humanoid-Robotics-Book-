from fastapi import FastAPI
from app.config import settings
from app.api.middleware.cors import add_cors_middleware

# Create FastAPI app instance
app = FastAPI(
    title="RAG Chatbot API",
    description="Retrieval-Augmented Generation API for Physical AI & Humanoid Robotics Book",
    version="1.0.0"
)

# Add CORS middleware
add_cors_middleware(app)

# Import and include routes after app creation to avoid circular imports
from app.api.routes import router as api_router
app.include_router(api_router, prefix="/api/v1")

@app.get("/")
def read_root():
    return {"message": "RAG Chatbot API is running!"}