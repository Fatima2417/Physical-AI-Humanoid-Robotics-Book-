from typing import List, Dict
import sys
import os
import uuid

# Add the backend path to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'backend'))

from app.db.qdrant import qdrant_service

class VectorLoader:
    def __init__(self):
        self.qdrant = qdrant_service
        # Ensure the collection exists
        self.qdrant.create_collection()

    def load_chunks_to_qdrant(self, chunks: List[Dict]) -> List[str]:
        """Load document chunks to Qdrant vector database"""
        point_ids = []

        for chunk in chunks:
            # Create metadata for Qdrant payload
            metadata = {
                'document_id': chunk.get('document_id', ''),
                'section_title': chunk.get('section_title', ''),
                'section_path': chunk.get('section_path', ''),
                'chunk_order': chunk.get('chunk_order', 0),
                'content': chunk.get('content', '')[:1000]  # Store a preview of content
            }

            # Add any additional metadata from the chunk
            if 'metadata' in chunk:
                metadata.update(chunk['metadata'])

            # Store the chunk in Qdrant with the actual embedding
            point_id = self.qdrant.store_embedding(
                text=chunk['content'],
                embedding=chunk['embedding'],
                doc_id=chunk.get('document_id', ''),
                metadata=metadata
            )

            point_ids.append(point_id)

        return point_ids

    def load_document_to_qdrant(self, doc: Dict) -> List[str]:
        """Load a single document (with chunking and embedding) to Qdrant"""
        from ..processors.embedding_generator import embedding_generator

        # Process document through chunking and embedding
        chunk_embeddings = embedding_generator.process_document_for_embeddings(doc)

        # Load the chunk embeddings to Qdrant
        point_ids = self.load_chunks_to_qdrant(chunk_embeddings)

        return point_ids

    def load_documents_to_qdrant(self, docs: List[Dict]) -> Dict[str, List[str]]:
        """Load multiple documents to Qdrant"""
        results = {}

        for doc in docs:
            file_path = doc.get('file_path', 'unknown')
            try:
                point_ids = self.load_document_to_qdrant(doc)
                results[file_path] = point_ids
            except Exception as e:
                print(f"Error loading document {file_path} to Qdrant: {str(e)}")
                results[file_path] = []

        return results

# Global instance
vector_loader = VectorLoader()