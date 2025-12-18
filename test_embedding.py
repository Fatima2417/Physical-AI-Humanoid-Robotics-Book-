#!/usr/bin/env python3
"""
Simple test script to verify embedding generation works with the fixed OpenRouter service
"""

import sys
from pathlib import Path

# Add paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "backend"))

# Load environment
from dotenv import load_dotenv
load_dotenv(project_root / "backend" / ".env")

from backend.app.rag.llm_service import openrouter_service

def test_embedding():
    print("Testing embedding generation...")

    test_text = "This is a test sentence for embedding."

    try:
        print(f"Generating embedding for: '{test_text}'")
        embedding = openrouter_service.get_embedding(test_text)

        print(f"Success! Generated embedding with {len(embedding)} dimensions")
        print(f"First 10 dimensions: {embedding[:10]}")

        # Test batch embedding
        print("\nTesting batch embedding...")
        texts = ["First test sentence", "Second test sentence", "Third test sentence"]
        embeddings = openrouter_service.get_embeddings_batch(texts)

        print(f"Success! Generated {len(embeddings)} embeddings")
        for i, emb in enumerate(embeddings):
            print(f"Text {i+1}: {len(emb)} dimensions, first 5: {emb[:5]}")

        return True

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_embedding()
    if success:
        print("\nSUCCESS: Embedding test passed!")
    else:
        print("\n✗ Embedding test failed!")
        sys.exit(1)