#!/usr/bin/env python3
"""
Script to run the ingestion process for creating embeddings
"""

import os
import sys
import argparse
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Add the backend path
backend_path = project_root / "backend"
sys.path.insert(0, str(backend_path))

# Load environment variables
from dotenv import load_dotenv
load_dotenv(backend_path / ".env")

# Now import the backend configuration to ensure it loads
from backend.app.config import settings

# Add ingestion path
ingestion_path = project_root / "ingestion"
sys.path.insert(0, str(ingestion_path))

def main():
    parser = argparse.ArgumentParser(description='Ingest Docusaurus content into RAG system')
    parser.add_argument('--source-path', type=str, required=True,
                        help='Path to the Docusaurus docs directory')
    parser.add_argument('--chunk-size', type=int, default=512,
                        help='Size of text chunks in tokens (default: 512)')
    parser.add_argument('--chunk-overlap', type=int, default=50,
                        help='Overlap between chunks in tokens (default: 50)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be processed without actually processing')

    args = parser.parse_args()

    # Import after paths are set
    from ingestion.src.parsers.markdown_parser import markdown_parser
    from ingestion.src.loaders.vector_loader import vector_loader

    # Validate source path
    source_path = Path(args.source_path)
    if not source_path.exists():
        print(f"Error: Source path {args.source_path} does not exist")
        sys.exit(1)

    if not source_path.is_dir():
        print(f"Error: Source path {args.source_path} is not a directory")
        sys.exit(1)

    print(f"Starting content ingestion from: {args.source_path}")
    print(f"Chunk size: {args.chunk_size}, Chunk overlap: {args.chunk_overlap}")
    print(f"Dry run: {args.dry_run}")
    print("-" * 50)

    # Parse all markdown files in the source directory
    print("Parsing markdown files...")
    docs = markdown_parser.parse_directory(str(source_path))
    print(f"Found {len(docs)} documents to process")

    if args.dry_run:
        print("DRY RUN: Would process the following documents:")
        for doc in docs:
            print(f"  - {doc['file_path']} ({len(doc.get('sections', []))} sections, {doc.get('word_count', 0)} words)")
        print("DRY RUN: Ingestion completed (no actual changes made)")
        return

    # Process each document
    total_chunks = 0
    for i, doc in enumerate(docs, 1):
        print(f"Processing ({i}/{len(docs)}): {doc['file_path']}")

        try:
            # Load document to Qdrant (this handles chunking and embedding internally)
            point_ids = vector_loader.load_document_to_qdrant(doc)
            num_chunks = len(point_ids)
            total_chunks += num_chunks
            print(f"  -> Stored {num_chunks} chunks in vector database")
        except Exception as e:
            print(f"  -> ERROR processing {doc['file_path']}: {str(e)}")

    print("-" * 50)
    print(f"Ingestion completed!")
    print(f"Total documents processed: {len(docs)}")
    print(f"Total chunks stored: {total_chunks}")
    print("Content is now available for RAG queries.")

if __name__ == "__main__":
    main()