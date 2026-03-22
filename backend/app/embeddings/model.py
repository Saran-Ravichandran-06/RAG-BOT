from sentence_transformers import SentenceTransformer
from app.core.config import settings

class EmbeddingModel:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            import torch
            import time
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"[{time.strftime('%H:%M:%S')}] Loading Embedding Model ({settings.EMBEDDING_MODEL}) on {device}...")
            start = time.time()
            cls._instance = SentenceTransformer(settings.EMBEDDING_MODEL, device=device)
            print(f"[{time.strftime('%H:%M:%S')}] Embedding Model Loaded in {time.time()-start:.2f}s")
        return cls._instance

    @classmethod
    def generate(cls, texts: list[str] | str):
        model = cls.get_instance()
        return model.encode(texts, convert_to_numpy=True)
