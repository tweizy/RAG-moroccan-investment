import cohere
import numpy as np
from sentence_transformers import SentenceTransformer
import streamlit as st

from vectordb.faiss_index import FaissIndex
from vectordb.metadata_store import retrieve_metadata

###############################################################################
#                          CONFIGURATION / CONSTANTS
###############################################################################

COHERE_API_KEY = "2RoQeyk7U6YzpaIGxlKJtNcqUkYHYn1boOSQEGmE"

FAISS_INDEX_PATH = "./vectordb/faiss_index.index"
FAISS_ID_MAP_PATH = "./vectordb/faiss_id_map.txt"
METADATA_DB_PATH = "./vectordb/metadata.db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"   
TOP_K = 5                                   
COHERE_MODEL = "command-xlarge-nightly"     
MAX_TOKENS = 1000                           
TEMPERATURE = 0.3                        

###############################################################################
#                    FUNCTION TO RETRIEVE CONTEXT FROM VECTOR DB
###############################################################################
def retrieve_context_from_db(user_query: str, top_k: int = TOP_K):
    """
    1) Embed user query with the same model used for chunk embeddings
    2) Load the FAISS index and search for top_k relevant chunks
    3) Retrieve chunk text + metadata from SQLite
    4) Return the retrieved chunks
    """
    # Load the embedding model
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    query_embedding = embedding_model.encode([user_query])[0]  # shape: (384,)

    # Load FAISS index
    faiss_index = FaissIndex(dimension=384)
    faiss_index.load(FAISS_INDEX_PATH, FAISS_ID_MAP_PATH)

    # Perform similarity search
    results = faiss_index.search(np.array(query_embedding), top_k=top_k)

    # For each chunk_id, retrieve metadata from SQLite
    retrieved_chunks = []
    for chunk_id, distance in results:
        metadata = retrieve_metadata(chunk_id, db_path=METADATA_DB_PATH)
        if metadata:
            snippet = metadata["chunk_text"]
            pdf_name = metadata["pdf_name"]
            retrieved_chunks.append({
                "chunk_id": chunk_id,
                "pdf_name": pdf_name,
                "chunk_text": snippet,
                "distance": distance
            })
    return retrieved_chunks


###############################################################################
#                  FUNCTION TO BUILD A PROMPT FOR COHERE
###############################################################################
def build_cohere_prompt(retrieved_chunks, user_query):
    """
    Constructs a robust, context-aware prompt for Cohere,
    combining instructions, retrieved chunks, and the user query.
    """

    # 1) System instructions: Tells the LLM how to behave
    system_instructions = (
        "You are an AI assistant specialized in providing well-structured, "
        "accurate, and concise answers about Moroccan investment insights. "
        "You have access to the following context from various PDFs. "
        "Use ONLY the context below to answer the user's query. "
        "If the context is insufficient to answer, mention it"
        "and then proceed to answer the question without taking into consideration the context. "
        "When providing information, include statistics only when you are certain of their accuracy "
        "and cite your sources whenever possible."
    )

    # 2) Gather relevant context from retrieved chunks
    context_lines = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        context_lines.append(
            f"[Context Chunk {i} | PDF: {chunk['pdf_name']}]"
            f"\n{chunk['chunk_text']}\n"
        )
    combined_context = "\n\n".join(context_lines)

    # 3) Final prompt construction
    #    Format it so Cohere has instructions -> context -> user query -> "Answer:"
    prompt = f"""
{system_instructions}

=====================
Context Provided:
=====================
{combined_context}

=====================
User Query:
=====================
{user_query}

=====================
Answer:
""".strip()

    return prompt


###############################################################################
#                     FUNCTION TO CALL COHERE FOR A RESPONSE
###############################################################################
def cohere_rag_response(user_query: str):
    """
    1) Retrieves context from the vector DB
    2) Builds a robust prompt for Cohere
    3) Calls Cohere with the prompt
    4) Returns the generated answer
    """

    # Step 1: Retrieve top_k relevant chunks from your vector DB
    retrieved_chunks = retrieve_context_from_db(user_query=user_query, top_k=TOP_K)

    # Step 2: Build the prompt
    prompt = build_cohere_prompt(retrieved_chunks, user_query)

    # Step 3: Call Cohere
    co = cohere.Client(COHERE_API_KEY)
    response = co.generate(
        model=COHERE_MODEL,
        prompt=prompt,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        stop_sequences=["---", "==="],    # Stop generating if we see these tokens
    )

    # Step 4: Return the generation text
    generated_text = response.generations[0].text.strip()
    return generated_text, retrieved_chunks


###############################################################################
#                            MAIN / TESTING
###############################################################################
if __name__ == "__main__":
    # Example user query
    user_query = "What incentives does Morocco offer to foreign automotive investors?"

    # Generate the final answer from Cohere
    final_answer, retrieved = cohere_rag_response(user_query)
    
    print("\n============ RETRIEVED CHUNKS ============")
    for i, chunk in enumerate(retrieved, start=1):
        snippet = chunk["chunk_text"][:200].replace('\n', ' ')
        print(f"\nChunk {i}:")
        print(f"  Chunk ID: {chunk['chunk_id']}")
        print(f"  PDF Name: {chunk['pdf_name']}")
        print(f"  Distance: {chunk['distance']:.4f}")
        print(f"  Snippet: {snippet}...")

    print("\n============ FINAL ANSWER ============")
    print(final_answer)