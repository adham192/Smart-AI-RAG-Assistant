"""
Vector Store Module (FAISS)
Handles:
- Creating FAISS index
- Adding embedded document chunks
- Similarity search
- Saving and loading index
"""

from typing import List, Dict, Tuple, Optional
import os
import faiss
import numpy as np
import pickle

from src.embedding import EmbeddingGenerator
from src.config import config


class FAISSVectorStore:
    """Manages FAISS vector store for the RAG system"""

    def __init__(self):
        """Initialize FAISS vector store"""
        self.embedding_generator = EmbeddingGenerator()
        self.dimension = self.embedding_generator.get_embedding_dimension()

        # Create FAISS index (L2 distance)
        self.index = faiss.IndexFlatL2(self.dimension)

        # Store metadata separately
        self.documents: List[Dict] = []


    # Create Vector Store
    def create_index(self, chunks: List[Dict]) -> None:
        """
        Create FAISS index from document chunks

        Args:
            chunks: List of chunk dictionaries
        """
        embeddings = []

        for chunk in chunks:
            vector = self.embedding_generator.embed_document(chunk["text"])
            embeddings.append(vector)
            self.documents.append(chunk)

        if embeddings:
            embeddings_array = np.array(embeddings).astype("float32")
            self.index.add(embeddings_array)


    # Add New Documents
    def add_documents(self, new_chunks: List[Dict]) -> None:
        """
        Add new chunks to existing index
        """
        embeddings = []

        for chunk in new_chunks:
            vector = self.embedding_generator.embed_document(chunk["text"])
            embeddings.append(vector)
            self.documents.append(chunk)

        if embeddings:
            embeddings_array = np.array(embeddings).astype("float32")
            self.index.add(embeddings_array)


    # Search
    def similarity_search(
        self,
        query: str,
        k: Optional[int] = None
    ) -> List[Dict]:
        """
        Perform similarity search

        Args:
            query: Search query
            k: Number of top results

        Returns:
            List of relevant document chunks
        """
        if k is None:
            k = config.TOP_K_RETRIEVAL

        query_vector = self.embedding_generator.embed_query(query)
        query_array = np.array([query_vector]).astype("float32")

        _, indices = self.index.search(query_array, k)

        results = []
        for idx in indices[0]:
            if idx < len(self.documents):
                results.append(self.documents[idx])

        return results


    # Search With Scores
    def similarity_search_with_scores(
        self,
        query: str,
        k: Optional[int] = None
    ) -> List[Tuple[Dict, float]]:
        """
        Perform similarity search with distance scores
        """
        if k is None:
            k = config.TOP_K_RETRIEVAL

        query_vector = self.embedding_generator.embed_query(query)
        query_array = np.array([query_vector]).astype("float32")

        distances, indices = self.index.search(query_array, k)

        results = []
        for idx, score in zip(indices[0], distances[0]):
            if idx < len(self.documents):
                results.append((self.documents[idx], float(score)))

        return results


    # Save & Load
    def save(self, path: str = "faiss_index") -> None:
        """
        Save FAISS index and metadata
        """
        if not os.path.exists(path):
            os.makedirs(path)

        faiss.write_index(self.index, os.path.join(path, "index.faiss"))

        with open(os.path.join(path, "documents.pkl"), "wb") as f:
            pickle.dump(self.documents, f)

    def load(self, path: str = "faiss_index") -> None:
        """
        Load FAISS index and metadata
        """
        self.index = faiss.read_index(os.path.join(path, "index.faiss"))

        with open(os.path.join(path, "documents.pkl"), "rb") as f:
            self.documents = pickle.load(f)


    # Info
    def get_info(self) -> Dict:
        """
        Get vector store statistics
        """
        return {
            "total_vectors": self.index.ntotal,
            "embedding_dimension": self.dimension
        }
