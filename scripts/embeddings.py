import json
import os
import sys

from sentence_transformers import SentenceTransformer


def load_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    """
    Loads a SentenceTransformer model for generating embeddings.

    :param model_name: Name of the SentenceTransformer model to load.
    :return: An instance of SentenceTransformer.
    """
    model = SentenceTransformer(model_name)
    return model


def embed_chunks(chunks, model):
    """
    Generates embeddings for each chunk of text using the provided SentenceTransformer model.

    :param chunks: List of dictionaries, each containing a 'chunk_text' key.
    :param model: A pre-loaded SentenceTransformer model.
    :return: List of dictionaries with an added 'embedding' key for each chunk.
    """
    # Extract all chunk texts for embedding
    texts = [chunk["chunk_text"] for chunk in chunks]
    
    # Generate embeddings for all texts at once with a progress bar
    embeddings = model.encode(texts, show_progress_bar=True)

    # Assign the generated embeddings back to their respective chunks
    for i, chunk in enumerate(chunks):
        chunk["embedding"] = embeddings[i].tolist()  # Convert numpy array to list for JSON compatibility

    return chunks


def main():
    """
    Main function to process chunks by generating embeddings and saving the results.

    Steps:
    1. Load the chunked data from a JSON file.
    2. Load the SentenceTransformer embedding model.
    3. Generate embeddings for each chunk.
    4. Save the embedded chunks to a new JSON file.
    """
    # Define paths for input and output files
    chunked_data_path = "./data/chunks/chunked_data.json"
    output_path = "./data/chunks/embedded_chunks.json"
    model_name = "all-MiniLM-L6-v2"

    # Check if the chunked data file exists
    if not os.path.exists(chunked_data_path):
        print(f"[ERROR] chunked_data.json not found at: {chunked_data_path}")
        sys.exit(1)

    # Step 1: Load chunked data from JSON file
    with open(chunked_data_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"[INFO] Loaded {len(chunks)} chunks from '{chunked_data_path}'.")

    # Step 2: Load the SentenceTransformer embedding model
    print(f"[INFO] Loading embedding model: '{model_name}' ...")
    model = load_embedding_model(model_name=model_name)

    # Step 3: Generate embeddings for each chunk
    print("[INFO] Generating embeddings for each chunk...")
    embedded_chunks = embed_chunks(chunks, model)
    print("[INFO] Embedding completed.")

    # Step 4: Save the embedded chunks to a new JSON file
    with open(output_path, "w", encoding="utf-8") as out_f:
        json.dump(embedded_chunks, out_f, indent=2)
    print(f"[INFO] Embedded chunks saved to '{output_path}'.")


if __name__ == "__main__":
    main()