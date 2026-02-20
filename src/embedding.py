"""
Embedding Module
Generate embeddings using Google's Gemini embedding model (google-genai SDK)
"""
from typing import List
from google import genai
from google.genai import types
from src.config import config


class EmbeddingGenerator:
    """Generate embeddings using Google Gemini API"""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or config.GOOGLE_API_KEY
        if not self.api_key:
            raise ValueError("Google API key is required. Set GOOGLE_API_KEY in your .env file.")

        self.client = genai.Client(api_key=self.api_key)
        self.model_name = config.EMBEDDING_MODEL

        # Auto-detect real dimension — gemini-embedding-001 returns 3072
        test = self.client.models.embed_content(
            model=self.model_name,
            contents="test",
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
        )
        self._dimension = len(test.embeddings[0].values)
        print(f"Embedding dimension detected: {self._dimension}")

    def embed_query(self, query: str) -> List[float]:
        """Embed a search query"""
        try:
            result = self.client.models.embed_content(
                model=self.model_name,
                contents=query,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
            )
            return result.embeddings[0].values
        except Exception as e:
            print(f"Error embedding query: {str(e)}")
            raise

    def embed_document(self, document: str) -> List[float]:
        """Embed a document chunk"""
        try:
            result = self.client.models.embed_content(
                model=self.model_name,
                contents=document,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
            )
            return result.embeddings[0].values
        except Exception as e:
            print(f"Error embedding document: {str(e)}")
            return [0.0] * self._dimension

    def get_embedding_dimension(self) -> int:
        """Returns the actual dimension from the model"""
        return self._dimension
