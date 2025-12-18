import asyncio
from typing import List, Dict, Optional
from app.rag.llm_service import openrouter_service
from app.rag.retrieval_service import retrieval_service
from app.rag.schemas import Citation
from app.config import settings
import time
import uuid

class RAGService:
    def __init__(self):
        self.llm_service = openrouter_service
        self.retrieval_service = retrieval_service

    async def process_global_query(
        self,
        query_text: str,
        session_id: Optional[str] = None,
        include_citations: bool = True
    ) -> Dict:
        """Process a global query against the entire book corpus"""
        start_time = time.time()

        # Retrieve relevant chunks using the retrieval service
        search_results = self.retrieval_service.retrieve_relevant_chunks(
            query_text=query_text,
            top_k=settings.top_k_chunks
        )

        if not search_results:
            return {
                "response_id": str(uuid.uuid4()),
                "answer": "Not found in book content.",
                "citations": [],
                "confidence_score": 0.0
            }

        # Assemble context from retrieved chunks
        context = self._assemble_context(search_results)

        # Generate response using LLM
        system_prompt = (
            "You are a helpful assistant for the Physical AI & Humanoid Robotics book. "
            "Answer the user's question based ONLY on the provided context. "
            "If the answer is not found in the provided context, respond with 'Not found in book content.' "
            "Do not use any external knowledge or make up information."
        )

        full_prompt = f"Context: {context}\n\nQuestion: {query_text}\n\nAnswer:"

        response_text = self.llm_service.generate_response(
            prompt=full_prompt,
            system_prompt=system_prompt
        )

        # Validate response is grounded in context
        is_valid = self.llm_service.validate_response(response_text, context)
        confidence_score = 0.9 if is_valid else 0.1

        # Create citations if requested
        citations = []
        if include_citations:
            citations = self._create_citations(search_results, query_text, response_text)

        processing_time = time.time() - start_time

        return {
            "response_id": str(uuid.uuid4()),
            "answer": response_text,
            "citations": citations,
            "confidence_score": confidence_score,
            "processing_time": processing_time
        }

    async def process_selected_text_query(
        self,
        query_text: str,
        selected_text: str,
        page_context: str = "",
        session_id: Optional[str] = None
    ) -> Dict:
        """Process a query using only the selected text as context"""
        start_time = time.time()

        # Use only the selected text as context - no retrieval from database
        context = selected_text

        # Generate response using only the selected text
        system_prompt = (
            "You are a helpful assistant for the Physical AI & Humanoid Robotics book. "
            "Answer the user's question based ONLY on the provided selected text. "
            "Do not use any external knowledge or information from outside the selected text. "
            "If the answer is not found in the provided text, respond with 'Not found in selected text.'"
        )

        full_prompt = f"Selected text: {context}\n\nQuestion: {query_text}\n\nAnswer:"

        response_text = self.llm_service.generate_response(
            prompt=full_prompt,
            system_prompt=system_prompt
        )

        # Validate response is grounded in context
        is_valid = self.llm_service.validate_response(response_text, context)
        confidence_score = 0.9 if is_valid else 0.1

        # Create a citation for the selected text
        citation = Citation(
            id=str(uuid.uuid4()),
            response_id=str(uuid.uuid4()),
            section_title="Selected Text",
            section_path=page_context or "unknown",
            text_snippet=selected_text[:200] + "..." if len(selected_text) > 200 else selected_text,
            similarity_score=1.0,
            page_number=None
        )

        processing_time = time.time() - start_time

        return {
            "response_id": str(uuid.uuid4()),
            "answer": response_text,
            "citations": [citation],
            "confidence_score": confidence_score,
            "processing_time": processing_time
        }

    def _assemble_context(self, search_results: List[Dict]) -> str:
        """Assemble context from search results"""
        context_parts = []

        for result in search_results:
            text = result.get("text", "")
            section_title = result.get("metadata", {}).get("section_title", "Unknown Section")
            score = result.get("score", 0)

            # Only include results with reasonable similarity score
            if score > 0.3:  # Adjust threshold as needed
                context_parts.append(f"Section: {section_title}\nContent: {text}\n---")

        return "\n".join(context_parts)

    def _create_citations(self, search_results: List[Dict], query: str, response: str) -> List[Citation]:
        """Create citation objects from search results"""
        citations = []

        for result in search_results:
            # Calculate relevance score based on similarity and content matching
            similarity_score = result.get("score", 0.0)

            # Create a citation object
            citation = Citation(
                id=str(uuid.uuid4()),
                response_id="",  # Will be set by caller
                section_title=result.get("metadata", {}).get("section_title", "Unknown Section"),
                section_path=result.get("metadata", {}).get("section_path", ""),
                text_snippet=result.get("text", "")[:200] + "..." if len(result.get("text", "")) > 200 else result.get("text", ""),
                similarity_score=min(1.0, similarity_score),  # Ensure score is between 0 and 1
                page_number=None
            )

            citations.append(citation)

        return citations

# Global instance
rag_service = RAGService()