from qdrant_client import QdrantClient
from qdrant_client.http import models
from typing import List, Dict, Optional
import uuid
from app.config import settings

class QdrantService:
    def __init__(self):
        self.client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            prefer_grpc=False
        )
        self.collection_name = settings.qdrant_collection

    def create_collection(self):
        """Create the collection if it doesn't exist"""
        try:
            self.client.get_collection(self.collection_name)
        except:
            # Collection doesn't exist, create it
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=1536,  # Default OpenAI embedding size, adjust based on model
                    distance=models.Distance.COSINE
                )
            )

    def store_embedding(self, text: str, embedding: List[float], doc_id: str, metadata: Dict) -> str:
        """Store a text embedding in Qdrant"""
        point_id = str(uuid.uuid4())

        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "text": text,
                        "doc_id": doc_id,
                        **metadata
                    }
                )
            ]
        )

        return point_id

    def search_similar(self, query_embedding: List[float], top_k: int = 5) -> List[Dict]:
        """Search for similar text chunks"""
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=top_k,
            with_payload=True
        )

        return [
            {
                "id": result.id,
                "text": result.payload["text"],
                "doc_id": result.payload["doc_id"],
                "metadata": {k: v for k, v in result.payload.items() if k not in ["text", "doc_id"]},
                "score": result.score
            }
            for result in results.points
        ]

    def get_all_points(self) -> List[Dict]:
        """Get all stored points (for debugging)"""
        results = self.client.scroll(
            collection_name=self.collection_name,
            with_payload=True,
            limit=10000  # Adjust as needed
        )

        points = []
        # Scroll returns a tuple of (points, next_offset), so we only iterate through the points
        for point in results[0]:  # results is (points, next_offset)
            points.append({
                "id": point.id,
                "text": point.payload["text"],
                "doc_id": point.payload["doc_id"],
                "metadata": {k: v for k, v in point.payload.items() if k not in ["text", "doc_id"]}
            })

        return points

# Global instance
qdrant_service = QdrantService()