from typing import List, Dict, Optional
from app.db.qdrant import qdrant_service
from app.rag.embedding_service import embedding_service
from app.config import settings

class RetrievalService:
    def __init__(self):
        self.qdrant_service = qdrant_service
        self.embedding_service = embedding_service

    def retrieve_relevant_chunks(self, query_text: str, top_k: Optional[int] = None) -> List[Dict]:
        """Retrieve the most relevant chunks for a given query"""
        if top_k is None:
            top_k = settings.top_k_chunks

        # Generate embedding for the query
        query_embedding = self.embedding_service.generate_embedding(query_text)

        # Search for relevant chunks in Qdrant
        search_results = self.qdrant_service.search_similar(
            query_embedding=query_embedding,
            top_k=top_k
        )

        return search_results

    def retrieve_by_document_id(self, document_id: str) -> List[Dict]:
        """Retrieve all chunks for a specific document"""
        # This would be implemented if Qdrant supported filtering by document_id
        # For now, return empty list - in a real implementation this would search by metadata
        all_chunks = self.qdrant_service.get_all_points()
        return [chunk for chunk in all_chunks if chunk.get("doc_id") == document_id]

    def validate_retrieval(self, query: str, results: List[Dict]) -> bool:
        """Validate that retrieved results are relevant to the query"""
        if not results:
            return False

        # Simple validation: check if any result has a reasonable similarity score
        for result in results:
            if result.get("score", 0) > 0.3:  # Threshold for relevance
                return True

        return False

# Global instance
retrieval_service = RetrievalService()