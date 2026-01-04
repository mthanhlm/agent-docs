"""Retrieval-Augmented Generation (RAG) system with vector similarity search."""

import os
import json
import logging
from typing import List, Dict, Any, Optional
import numpy as np
import tiktoken
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class Document:
    """Represents a knowledge chunk with content and metadata."""
    
    def __init__(self, content: str, metadata: Optional[Dict[str, Any]] = None):
        self.content = content
        self.metadata = metadata or {}

    def to_dict(self) -> dict:
        """Serialize document to dictionary format."""
        return {"content": self.content, "metadata": self.metadata}


class RecursiveCharacterTextSplitter:
    """Splits text into chunks based on token count with overlap."""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self.tokenizer = None

    def split_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        if not self.tokenizer:
            return [
                text[i:i + self.chunk_size] 
                for i in range(0, len(text), self.chunk_size - self.chunk_overlap)
            ]
        
        tokens = self.tokenizer.encode(text)
        chunks = []
        for i in range(0, len(tokens), self.chunk_size - self.chunk_overlap):
            chunk_tokens = tokens[i:i + self.chunk_size]
            chunks.append(self.tokenizer.decode(chunk_tokens))
        return chunks


class SimpleVectorStore:
    """Persistent vector store with JSON-based storage."""
    
    def __init__(self, storage_path: str = "vector_store.json"):
        self.storage_path = storage_path
        self.vectors: List[np.ndarray] = []
        self.documents: List[Document] = []

    def add_documents(self, vectors: List[np.ndarray], documents: List[Document]):
        """Add new documents with their embeddings."""
        self.vectors.extend(vectors)
        self.documents.extend(documents)
        self.save_to_json()

    def similarity_search(self, query_vector: np.ndarray, k: int = 3, threshold: float = 0.3) -> List[tuple]:
        """Find documents similar to the query vector using cosine similarity."""
        if not self.vectors:
            return []
        
        vector_matrix = np.array(self.vectors)
        dot_product = np.dot(vector_matrix, query_vector)
        norms = np.linalg.norm(vector_matrix, axis=1) * np.linalg.norm(query_vector)
        similarities = dot_product / (norms + 1e-9)
        
        results = []
        for i, score in enumerate(similarities):
            if score >= threshold:
                results.append((score, self.documents[i]))
                logger.info(f"RAG Match: Score={score:.4f}, Content={self.documents[i].content[:50]}...")
        
        results.sort(key=lambda x: x[0], reverse=True)
        return results[:k]

    def save_to_json(self):
        """Persist store to JSON file."""
        data = {
            "documents": [doc.to_dict() for doc in self.documents],
            "vectors": [v.tolist() for v in self.vectors]
        }
        with open(self.storage_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_from_json(self):
        """Load store from JSON file."""
        if not os.path.exists(self.storage_path):
            logger.info(f"RAG: {self.storage_path} not found. Initializing new storage.")
            self.save_to_json()
            return

        try:
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.documents = [Document(d["content"], d["metadata"]) for d in data["documents"]]
                self.vectors = [np.array(v) for v in data["vectors"]]
            logger.info(f"RAG: Loaded {len(self.documents)} documents from {self.storage_path}")
        except Exception as e:
            logger.error(f"RAG: Error loading JSON: {e}")


class RAGSystem:
    """Main RAG pipeline combining embedding, storage, and retrieval."""
    
    def __init__(
        self,
        threshold: float = 0.3,
        top_k: int = 3,
        storage_path: str = "vector_store.json",
        clear_history: bool = False,
        documents: Optional[List[str]] = None
    ):
        logger.info("Loading embedding model for RAG...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.splitter = RecursiveCharacterTextSplitter()
        self.vector_store = SimpleVectorStore(storage_path)
        
        if clear_history and os.path.exists(storage_path):
            os.remove(storage_path)
            logger.info("RAG: Cleared existing vector store.")
        else:
            self.vector_store.load_from_json()
            
        self.threshold = threshold
        self.top_k = top_k
        
        if documents:
            self.add_knowledge(documents)

    def add_knowledge(self, texts: List[str]):
        """Process and index new documents."""
        all_chunks = []
        for text in texts:
            chunks = self.splitter.split_text(text)
            for chunk in chunks:
                all_chunks.append(Document(content=chunk))
        
        if not all_chunks:
            return

        contents = [doc.content for doc in all_chunks]
        vectors = self.model.encode(contents, convert_to_numpy=True)
        self.vector_store.add_documents(list(vectors), all_chunks)

    def query(self, query_text: str, k: Optional[int] = None) -> str:
        """Retrieve relevant context for a query."""
        k = k or self.top_k
        query_vector = self.model.encode(query_text)
        results = self.vector_store.similarity_search(query_vector, k=k, threshold=self.threshold)
        
        if not results:
            return ""
        
        logger.info(f"RAG: Found {len(results)} relevant chunks.")
        context_parts = [doc.content for score, doc in results]
        return "\n---\n".join(context_parts)
