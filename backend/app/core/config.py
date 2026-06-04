import os
from pathlib import Path

class Settings:
    PROJECT_Title = "RAG Chatbot"
    PROJECT_VERSION = "1.0.0"
    
    # Paths
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    DATA_DIR = BASE_DIR / "data"
    CHATS_DIR = DATA_DIR / "chats"
    
    # RAG Settings
    CHUNK_SIZE = 600
    CHUNK_OVERLAP = 50
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    LLM_MODEL = "phi3:latest"
    OLLAMA_BASE_URL = "http://127.0.0.1:11434"
    
    # LLM Memory Optimization
    LLM_CONTEXT_LENGTH = 2048
    LLM_TEMPERATURE = 0.3
    LLM_KEEP_ALIVE = "5m"  # Keep model in memory for 5 minutes
    LLM_NUM_GPU = 32  # Offload all 32 layers for Phi3 to GPU (approx 2.3GB total)
    
    # Vector Search
    TOP_K = 3
    SIMILARITY_THRESHOLD = 0.3  # Filter out low relevance
    RETRIEVAL_CONFIDENCE_THRESHOLD = 0.45  # Skip LLM when best retrieved chunk is weak
    
    # Evaluation
    ENABLE_EVALUATION = False
    HALLUCINATION_THRESHOLD = 0.5
    
    def __init__(self):
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.CHATS_DIR.mkdir(parents=True, exist_ok=True)

settings = Settings()
