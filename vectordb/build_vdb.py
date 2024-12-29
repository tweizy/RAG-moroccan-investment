import os
import json
import numpy as np

from metadata_store import initialize_metadata_db, insert_metadata
from faiss_index import FaissIndex

def main():
    embedded_chunks_path = "./data/chunks/embedded_chunks.json"
    metadata_db_path = "./vectordb/metadata.db"
    faiss_index_path = "./vectordb/faiss_index.index"
    faiss_id_map_path = "./vectordb/faiss_id_map.txt"

    if not os.path.exists(embedded_chunks_path):
        print(f"[ERROR] Embedded chunks file not found at {embedded_chunks_path}")
        return

    # 1. Load embedded chunks
    with open(embedded_chunks_path, "r", encoding="utf-8") as f:
        embedded_chunks = json.load(f)
    print(f"[INFO] Loaded {len(embedded_chunks)} embedded chunks.")

    # 2. Initialize metadata DB
    initialize_metadata_db(db_path=metadata_db_path)
    print(f"[INFO] Initialized metadata database '{metadata_db_path}'.")

    # 3. Insert chunk metadata
    insert_metadata(embedded_chunks, db_path=metadata_db_path)
    print(f"[INFO] Metadata inserted into '{metadata_db_path}'.")

    # 4. Initialize FAISS index
    dimension = 384  # match your embedding model dimension
    faiss_index = FaissIndex(dimension=dimension, index_type="Flat")
    print(f"[INFO] Initialized FAISS index (dimension={dimension}).")

    # 5. Add vectors to FAISS
    vectors = []
    ids = []
    for chunk in embedded_chunks:
        # chunk["embedding"] is a list of floats -> convert to np.ndarray
        vectors.append(chunk["embedding"])
        ids.append(chunk["chunk_id"])

    vectors_np = np.array(vectors, dtype="float32")
    faiss_index.add_vectors(vectors_np, ids)
    print("[INFO] Added vectors to FAISS index.")

    # 6. Save FAISS index + ID map
    faiss_index.save(faiss_index_path, faiss_id_map_path)
    print(f"[INFO] FAISS index saved to '{faiss_index_path}'.")
    print(f"[INFO] ID map saved to '{faiss_id_map_path}'.")

    print("\n=== Vector DB Build Complete ===")

if __name__ == "__main__":
    main()