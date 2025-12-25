import os
import json
import logging
import numpy as np
import tiktoken
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class Document:
    """Đại diện cho một mẩu kiến thức."""
    def __init__(self, content: str, metadata: Optional[Dict[str, Any]] = None):
        self.content = content
        self.metadata = metadata or {}

    def to_dict(self):
        return {"content": self.content, "metadata": self.metadata}

class RecursiveCharacterTextSplitter:
    """Chia nhỏ văn bản sử dụng tiktoken để tính toán độ dài."""
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except:
            self.tokenizer = None

    def split_text(self, text: str) -> List[str]:
        if not self.tokenizer:
            # Fallback nếu không có tiktoken
            return [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size - self.chunk_overlap)]
        
        tokens = self.tokenizer.encode(text)
        chunks = []
        for i in range(0, len(tokens), self.chunk_size - self.chunk_overlap):
            chunk_tokens = tokens[i:i + self.chunk_size]
            chunks.append(self.tokenizer.decode(chunk_tokens))
        return chunks

class SimpleVectorStore:
    """Lưu trữ vector và tài liệu, hỗ trợ xuất/nhập JSON."""
    def __init__(self, storage_path: str = "vector_store.json"):
        self.storage_path = storage_path
        self.vectors: List[np.ndarray] = []
        self.documents: List[Document] = []

    def add_documents(self, vectors: List[np.ndarray], documents: List[Document]):
        self.vectors.extend(vectors)
        self.documents.extend(documents)
        self.save_to_json()

    def similarity_search(self, query_vector: np.ndarray, k: int = 3, threshold: float = 0.3) -> List[tuple]:
        if not self.vectors:
            return []
        
        vector_matrix = np.array(self.vectors)
        # Cosine similarity
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
        """Lưu toàn bộ tri thức và vector ra file JSON để dễ theo dõi."""
        data = {
            "documents": [doc.to_dict() for doc in self.documents],
            "vectors": [v.tolist() for v in self.vectors]
        }
        with open(self.storage_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_from_json(self):
        """Khôi phục tri thức từ file JSON."""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.documents = [Document(d["content"], d["metadata"]) for d in data["documents"]]
                    self.vectors = [np.array(v) for v in data["vectors"]]
                logger.info(f"RAG: Loaded {len(self.documents)} documents from {self.storage_path}")
            except Exception as e:
                logger.error(f"RAG: Error loading JSON: {e}")

class RAGSystem:
    """Hệ thống RAG chính kết hợp tất cả các thành phần."""
    def __init__(self, threshold: float = 0.3, top_k: int = 3, storage_path: str = "vector_store.json", clear_history: bool = False):
        logger.info("Loading HuggingFace model for RAG...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.splitter = RecursiveCharacterTextSplitter()
        self.vector_store = SimpleVectorStore(storage_path)
        
        if clear_history:
            if os.path.exists(storage_path):
                os.remove(storage_path)
            logger.info("RAG: cleared existing vector store.")
        else:
            self.vector_store.load_from_json()
            
        self.threshold = threshold
        self.top_k = top_k

    def add_knowledge(self, texts: List[str]):
        """Nạp thêm văn bản vào tri thức."""
        all_chunks = []
        for text in texts:
            chunks = self.splitter.split_text(text)
            for chunk in chunks:
                all_chunks.append(Document(content=chunk))
        
        if not all_chunks:
            return

        # Tạo embeddings hàng loạt bằng model HF
        contents = [doc.content for doc in all_chunks]
        vectors = self.model.encode(contents, convert_to_numpy=True)
        
        self.vector_store.add_documents(list(vectors), all_chunks)

    def query(self, query_text: str, k: Optional[int] = None) -> str:
        """Truy vấn tri thức liên quan."""
        k = k or self.top_k
        query_vector = self.model.encode(query_text)
        results = self.vector_store.similarity_search(query_vector, k=k, threshold=self.threshold)
        
        if not results:
            return ""
        
        logger.info(f"RAG: Found {len(results)} relevant chunks.")
        context_parts = [doc.content for score, doc in results]
        return "\n---\n".join(context_parts)
