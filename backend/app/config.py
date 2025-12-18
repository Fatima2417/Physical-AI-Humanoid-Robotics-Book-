from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    # OpenRouter configuration
    openrouter_api_key: str
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_chat_model: str = "qwen/qwen-2.5-72b-instruct"
    openrouter_embed_model: str = "qwen/qwen-2.5-embedding"

    # Qdrant configuration
    qdrant_url: str
    qdrant_api_key: str
    qdrant_collection: str = "book_embeddings"

    # Neon Postgres configuration
    neon_database_url: str

    # Application settings
    debug: bool = False
    max_query_length: int = 2000
    max_selected_text_length: int = 5000
    response_timeout: int = 30
    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k_chunks: int = 5

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Create settings instance
settings = Settings()

# Validate required environment variables are present
def validate_settings():
    required_vars = [
        'openrouter_api_key',
        'qdrant_url',
        'qdrant_api_key',
        'neon_database_url'
    ]

    missing_vars = []
    for var in required_vars:
        if not getattr(settings, var):
            missing_vars.append(var)

    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

# Validate settings on import
validate_settings()