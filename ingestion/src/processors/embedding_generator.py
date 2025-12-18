from typing import List, Dict
import sys
import os

# Add the backend path to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'backend'))

from app.rag.embedding_service import embedding_service

class EmbeddingGenerator:
    def __init__(self):
        self.embedding_service = embedding_service

    def generate_embeddings_for_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """Generate embeddings for a list of document chunks"""
        chunk_embeddings = []

        # Process in batches to be more efficient
        batch_size = 10  # Adjust based on API limits
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]

            # Extract text content for batch processing
            texts = [chunk['content'] for chunk in batch]

            # Generate embeddings for the batch
            embeddings = self.embedding_service.generate_embeddings_batch(texts)

            # Attach embeddings to chunks
            for j, chunk in enumerate(batch):
                chunk_with_embedding = chunk.copy()
                chunk_with_embedding['embedding'] = embeddings[j]
                chunk_embeddings.append(chunk_with_embedding)

        return chunk_embeddings

    def generate_single_embedding(self, text: str) -> List[float]:
        """Generate a single embedding for text"""
        return self.embedding_service.generate_embedding(text)

    def process_document_for_embeddings(self, doc: Dict) -> List[Dict]:
        """Process a document through chunking and embedding generation"""
        from ingestion.src.chunkers.document_chunker import document_chunker

        # First chunk the document
        chunks = document_chunker.chunk_document(doc)

        # Then generate embeddings for all chunks
        chunk_embeddings = self.generate_embeddings_for_chunks(chunks)

        return chunk_embeddings

# Global instance
embedding_generator = EmbeddingGenerator()