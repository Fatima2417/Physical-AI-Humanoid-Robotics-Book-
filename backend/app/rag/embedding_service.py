from typing import List
from app.rag.llm_service import openrouter_service

class EmbeddingService:
    def __init__(self):
        self.openrouter = openrouter_service

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        return self.openrouter.get_embedding(text)

    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        return self.openrouter.get_embeddings_batch(texts)

    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings"""
        # Calculate dot product
        dot_product = sum(a * b for a, b in zip(embedding1, embedding2))

        # Calculate magnitudes
        magnitude1 = sum(a * a for a in embedding1) ** 0.5
        magnitude2 = sum(b * b for b in embedding2) ** 0.5

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        # Calculate cosine similarity
        similarity = dot_product / (magnitude1 * magnitude2)

        # Ensure similarity is between 0 and 1
        return max(0.0, min(1.0, (similarity + 1) / 2))

embedding_service = EmbeddingService()