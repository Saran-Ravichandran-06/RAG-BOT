import faiss
import numpy as np
import pickle
import threading
import time
from app.core.config import settings

class FaissVectorStore:
    _store_cache = {}
    _cache_lock = threading.RLock()

    @classmethod
    def get_store(cls, chat_id: str):
        cache_start = time.time()
        with cls._cache_lock:
            if chat_id in cls._store_cache:
                print(
                    f"[{time.strftime('%H:%M:%S')}] FAISS cache hit for chat_id={chat_id} "
                    f"({time.time() - cache_start:.4f}s)"
                )
                return cls._store_cache[chat_id]

            store = cls(chat_id)
            cls._store_cache[chat_id] = store
            print(
                f"[{time.strftime('%H:%M:%S')}] FAISS cache miss/load for chat_id={chat_id} "
                f"({time.time() - cache_start:.4f}s)"
            )
            return store

    @classmethod
    def invalidate(cls, chat_id: str):
        with cls._cache_lock:
            cls._store_cache.pop(chat_id, None)

    def __init__(self, chat_id: str):
        self.chat_id = chat_id
        self.index_path = settings.CHATS_DIR / str(chat_id) / "index.faiss"
        self.meta_path = settings.CHATS_DIR / str(chat_id) / "metadata.pkl"
        self.dimension = 384  # Dimension for all-MiniLM-L6-v2
        self._lock = threading.RLock()

        # Ensure chat directory exists
        (settings.CHATS_DIR / str(chat_id)).mkdir(parents=True, exist_ok=True)

        load_start = time.time()
        self.index = self._load_index()
        self.metadata = self._load_metadata()
        print(
            f"[{time.strftime('%H:%M:%S')}] FAISS index/metadata loaded for chat_id={chat_id} "
            f"(vectors={self.index.ntotal}, metadata={len(self.metadata)}, "
            f"{time.time() - load_start:.4f}s)"
        )

    def _load_index(self):
        if self.index_path.exists():
            return faiss.read_index(str(self.index_path))
        return faiss.IndexFlatIP(self.dimension)  # Inner Product for cosine similarity (normalized vectors)

    def _load_metadata(self):
        if self.meta_path.exists():
            with open(self.meta_path, "rb") as f:
                return pickle.load(f)
        return []

    def save(self):
        with self._lock:
            faiss.write_index(self.index, str(self.index_path))
            with open(self.meta_path, "wb") as f:
                pickle.dump(self.metadata, f)

    def add_documents(self, embeddings: np.ndarray, chunks: list[str]):
        with self._lock:
            # Normalize for cosine similarity
            faiss.normalize_L2(embeddings)
            self.index.add(embeddings)
            self.metadata.extend(chunks)
            faiss.write_index(self.index, str(self.index_path))
            with open(self.meta_path, "wb") as f:
                pickle.dump(self.metadata, f)

    def search(self, query_vector: np.ndarray, k: int = settings.TOP_K) -> list[tuple[str, float]]:
        with self._lock:
            # Normalize query
            faiss.normalize_L2(query_vector)

            if self.index.ntotal == 0:
                return []

            scores, indices = self.index.search(query_vector, k)

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx != -1 and score >= settings.SIMILARITY_THRESHOLD:
                    if 0 <= idx < len(self.metadata):
                        results.append((self.metadata[idx], float(score)))
                    # Else: Index likely corrupted or out of sync with metadata

            return results
