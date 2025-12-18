from typing import List, Dict
import re

class DocumentChunker:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(self, doc: Dict) -> List[Dict]:
        """Chunk a parsed document into smaller pieces"""
        chunks = []
        doc_id = doc.get('file_path', 'unknown')
        title = doc.get('title', 'Unknown')
        sections = doc.get('sections', [])

        chunk_id = 0
        for section in sections:
            section_title = section.get('title', 'Section')
            section_content = section.get('content', '')

            section_chunks = self._chunk_text(
                section_content,
                doc_id,
                section_title,
                chunk_id
            )

            chunks.extend(section_chunks)
            chunk_id += len(section_chunks)

        return chunks

    def _chunk_text(self, text: str, doc_id: str, section_title: str, start_chunk_id: int = 0) -> List[Dict]:
        """Chunk a text into smaller pieces with overlap"""
        chunks = []

        # Split text into sentences to avoid breaking in the middle of sentences
        sentences = self._split_into_sentences(text)

        current_chunk = ""
        current_chunk_tokens = 0
        chunk_id = start_chunk_id

        for sentence in sentences:
            sentence_tokens = len(sentence.split())

            # If adding this sentence would exceed chunk size
            if current_chunk_tokens + sentence_tokens > self.chunk_size and current_chunk:
                # Save current chunk
                chunks.append({
                    'id': f"{doc_id}_chunk_{chunk_id}",
                    'document_id': doc_id,
                    'content': current_chunk.strip(),
                    'section_title': section_title,
                    'section_path': self._get_section_path(doc_id, section_title),
                    'chunk_order': chunk_id,
                    'metadata': {}
                })

                chunk_id += 1

                # Start new chunk with overlap
                if self.chunk_overlap > 0:
                    # Get the last few sentences from current chunk to use as overlap
                    overlap_sentences = self._get_overlap_sentences(current_chunk, self.chunk_overlap)
                    current_chunk = overlap_sentences + " " + sentence
                    current_chunk_tokens = len(current_chunk.split())
                else:
                    current_chunk = sentence
                    current_chunk_tokens = sentence_tokens
            else:
                # Add sentence to current chunk
                current_chunk += " " + sentence if current_chunk else sentence
                current_chunk_tokens += sentence_tokens

        # Add the last chunk if it has content
        if current_chunk.strip():
            chunks.append({
                'id': f"{doc_id}_chunk_{chunk_id}",
                'document_id': doc_id,
                'content': current_chunk.strip(),
                'section_title': section_title,
                'section_path': self._get_section_path(doc_id, section_title),
                'chunk_order': chunk_id,
                'metadata': {}
            })

        return chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # This is a simple sentence splitter
        # In a production implementation, use a more sophisticated NLP library
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences

    def _get_overlap_sentences(self, text: str, token_count: int) -> str:
        """Get the last few sentences from text that contain approximately token_count tokens"""
        sentences = self._split_into_sentences(text)
        overlap_sentences = []
        current_tokens = 0

        # Go backwards through sentences
        for sentence in reversed(sentences):
            sentence_tokens = len(sentence.split())
            if current_tokens + sentence_tokens <= token_count:
                overlap_sentences.insert(0, sentence)
                current_tokens += sentence_tokens
            else:
                break

        return " ".join(overlap_sentences)

    def _get_section_path(self, doc_id: str, section_title: str) -> str:
        """Generate a section path from document ID and section title"""
        # Convert file path to a more web-friendly path
        path = doc_id.replace('\\', '/').replace('.md', '').replace('.markdown', '')

        # If the path starts with common documentation folders, adjust accordingly
        if 'docs/' in path:
            path = path[path.find('docs/') + 5:]  # Remove 'docs/' prefix
        elif 'Book/docs/' in path:
            path = path[path.find('Book/docs/') + 10:]  # Remove 'Book/docs/' prefix

        # Add section title if it's not generic
        if section_title.lower() not in ['introduction', 'section', 'chapter']:
            path = f"{path}#{section_title.lower().replace(' ', '-')}"

        return f"/{path}" if not path.startswith('/') else path

# Global instance
document_chunker = DocumentChunker()