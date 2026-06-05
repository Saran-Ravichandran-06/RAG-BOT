from sentence_transformers import SentenceTransformer
from app.core.config import settings

class EmbeddingModel:
    _instance = None
    _device = None

    @classmethod
    def _select_device(cls):
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            import time
            cls._device = cls._select_device()
            print(f"[{time.strftime('%H:%M:%S')}] Loading Embedding Model on {cls._device}")
            print(f"[{time.strftime('%H:%M:%S')}] Embedding Model: {settings.EMBEDDING_MODEL}")
            start = time.time()
            cls._instance = SentenceTransformer(settings.EMBEDDING_MODEL, device=cls._device)
            print(f"[{time.strftime('%H:%M:%S')}] Embedding Model Loaded in {time.time()-start:.2f}s")
        return cls._instance

    @classmethod
    def get_device(cls):
        if cls._device is None:
            cls._device = cls._select_device()
        return cls._device

    @classmethod
    def generate(cls, texts: list[str] | str):
        model = cls.get_instance()
        return model.encode(texts, convert_to_numpy=True)
