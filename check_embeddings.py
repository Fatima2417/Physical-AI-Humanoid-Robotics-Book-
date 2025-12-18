#!/usr/bin/env python3
"""
Script to check if embeddings were successfully stored in Qdrant
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

from backend.app.db.qdrant import qdrant_service

def check_embeddings():
    print("Checking Qdrant collection status...")

    try:
        # Get collection info
        collection_info = qdrant_service.client.get_collection(qdrant_service.collection_name)
        print(f"Collection: {qdrant_service.collection_name}")
        print(f"Vectors count: {collection_info.points_count}")
        print(f"Status: {collection_info.status}")

        if collection_info.points_count > 0:
            print(f"\nSUCCESS: Successfully stored {collection_info.points_count} embeddings in Qdrant!")

            # Get a few sample points to verify
            print("\nSample points from the collection:")
            sample_points = qdrant_service.get_all_points()
            for i, point in enumerate(sample_points[:3]):  # Show first 3 points
                print(f"  Point {i+1}: ID={point['id'][:8]}..., Doc={point['doc_id']}, Text preview: {point['text'][:100]}...")

        else:
            print("\n! No embeddings found in the collection yet.")

        return collection_info.points_count > 0

    except Exception as e:
        print(f"Error checking collection: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = check_embeddings()
    if success:
        print("\nSUCCESS: Embeddings verification passed!")
    else:
        print("\nINFO: Checking status...")