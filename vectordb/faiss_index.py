import faiss
import numpy as np
from typing import List

class FaissIndex:
    def __init__(self, dimension=384, index_type="Flat"):
        """
        Initialize a FAISS index for similarity search.
        
        :param dimension: Dimension of the embedding vectors.
        :param index_type: Type of FAISS index ('Flat', 'HNSW', 'IVF', etc.).
        """
        self.dimension = dimension
        self.id_map = []
        if index_type == "Flat":
            self.index = faiss.IndexFlatL2(dimension)  # Basic L2 distance
        else:
            self.index = faiss.IndexFlatL2(dimension)

    def add_vectors(self, vectors: np.ndarray, ids: List[str]) -> None:
        """
        Adds vectors (embeddings) to the index.
        
        :param vectors: np.ndarray of shape (n, dimension) with float32 embeddings.
        :param ids: List of string IDs corresponding to each vector.
        """
        if not isinstance(vectors, np.ndarray):
            vectors = np.array(vectors).astype("float32")

        self.index.add(vectors)
        self.id_map.extend(ids)

    def search(self, query_vector: np.ndarray, top_k: int = 5):
        """
        Perform a similarity search on the index.
        
        :param query_vector: A single embedding vector shape (dimension,) or (1, dimension).
        :param top_k: Number of similar results to retrieve.
        :return: List of (id, distance) tuples.
        """
        if len(query_vector.shape) == 1:
            query_vector = np.expand_dims(query_vector, axis=0)

        distances, indices = self.index.search(query_vector.astype("float32"), top_k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.id_map):
                results.append((self.id_map[idx], float(dist)))
        return results

    def save(self, faiss_file: str, id_map_file: str):
        """
        Save the FAISS index and ID map to disk.
        
        :param faiss_file: Path to save the FAISS index file.
        :param id_map_file: Path to save the ID map file.
        """
        faiss.write_index(self.index, faiss_file)
        with open(id_map_file, "w", encoding="utf-8") as f:
            for _id in self.id_map:
                f.write(_id + "\n")

    def load(self, faiss_file: str, id_map_file: str):
        """
        Load the FAISS index and ID map from disk.
        
        :param faiss_file: Path to the FAISS index file.
        :param id_map_file: Path to the ID map file.
        """
        self.index = faiss.read_index(faiss_file)
        self.id_map = []
        with open(id_map_file, "r", encoding="utf-8") as f:
            for line in f:
                self.id_map.append(line.strip())