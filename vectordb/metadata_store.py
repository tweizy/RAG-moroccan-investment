import sqlite3
from typing import List, Dict, Any

def initialize_metadata_db(db_path: str = "metadata.db") -> None:
    """
    Creates (or verifies) a table for chunk metadata if it doesn't exist already.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS chunk_metadata (
        chunk_id TEXT PRIMARY KEY,
        pdf_name TEXT,
        chunk_text TEXT
    )
    """)
    conn.commit()
    conn.close()

def insert_metadata(chunks: List[Dict[str, Any]], db_path: str = "metadata.db") -> None:
    """
    Insert or update metadata for each chunk in the SQLite database.

    :param chunks: List of chunk dictionaries (with chunk_id, pdf_name, chunk_text, etc.).
    :param db_path: Path to the SQLite database file.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    for chunk in chunks:
        cur.execute("""
            INSERT OR REPLACE INTO chunk_metadata 
            (chunk_id, pdf_name, chunk_text)
            VALUES (?, ?, ?)
        """, (
            chunk["chunk_id"],
            chunk.get("pdf_name", ""),
            chunk.get("chunk_text", "")
        ))
    conn.commit()
    conn.close()

def retrieve_metadata(chunk_id: str, db_path: str = "metadata.db"):
    """
    Retrieve metadata for a specific chunk_id.

    :param chunk_id: The chunk_id to look up.
    :param db_path: Path to the SQLite database file.
    :return: A dict with metadata, or None if not found.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT chunk_id, pdf_name, chunk_text FROM chunk_metadata WHERE chunk_id = ?", (chunk_id,))
    row = cur.fetchone()
    conn.close()

    if row:
        metadata = {
            "chunk_id": row[0],
            "pdf_name": row[1],
            "chunk_text": row[2]
        }
        print(f"[INFO] Retrieved metadata for chunk_id '{chunk_id}'.")
        return metadata
    else:
        print(f"[WARN] No metadata found for chunk_id '{chunk_id}'.")
        return None