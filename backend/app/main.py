from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.api import chat, upload, query
from app.core.config import settings
from app.embeddings.model import EmbeddingModel
import requests
import time
import subprocess

def check_nvidia_smi():
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return True, "nvidia-smi is available. GPU detected."
        return False, "nvidia-smi returned non-zero exit code."
    except Exception as e:
        return False, f"nvidia-smi not found or failed: {str(e)}"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Preload the embedding model
    print(f"[{time.strftime('%H:%M:%S')}] Preloading Embedding Model ({settings.EMBEDDING_MODEL})...")
    EmbeddingModel.get_instance()
    
    # 2. Verify GPU Usage for Ollama
    print(f"[{time.strftime('%H:%M:%S')}] Verifying Ollama GPU usage...")
    has_gpu, gpu_msg = check_nvidia_smi()
    if has_gpu:
        print(f"[{time.strftime('%H:%M:%S')}] GPU Check: {gpu_msg}")
    else:
        print(f"[{time.strftime('%H:%M:%S')}] WARNING GPU Check: {gpu_msg} (Ollama might fallback to CPU)")

    try:
        url = f"{settings.OLLAMA_BASE_URL}/api/tags"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            print(f"[{time.strftime('%H:%M:%S')}] Ollama server is reachable.")
            print(f"[{time.strftime('%H:%M:%S')}] GPU Offloading is configured to use {settings.LLM_NUM_GPU} layers.")
            print(f"[{time.strftime('%H:%M:%S')}] TIP: To validate GPU usage during inference:")
            print(f"[{time.strftime('%H:%M:%S')}] 1. Run 'ollama ps' in terminal. Look for '100% GPU' under PROCESSOR.")
            print(f"[{time.strftime('%H:%M:%S')}] 2. Run 'nvidia-smi' to check VRAM utilization.")
            print(f"[{time.strftime('%H:%M:%S')}] NOTE: If Ollama runs on CPU, ensure you have the GPU-enabled version of Ollama installed.")
        else:
            print(f"[{time.strftime('%H:%M:%S')}] WARNING: Ollama server returned status code {response.status_code}")
    except requests.exceptions.RequestException:
        print(f"[{time.strftime('%H:%M:%S')}] WARNING: Could not connect to Ollama server at {settings.OLLAMA_BASE_URL}. Ensure it is running.")
        
    yield

app = FastAPI(title=settings.PROJECT_Title, version=settings.PROJECT_VERSION, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router, prefix="/chat", tags=["chat"])
app.include_router(upload.router, prefix="/chat", tags=["upload"])
app.include_router(query.router, prefix="/chat", tags=["query"])

@app.get("/chat/gpu-status", tags=["diagnostic"])
def get_gpu_status():
    """Diagnostic endpoint to check GPU settings and Ollama status."""
    has_gpu, gpu_msg = check_nvidia_smi()
    status = {
        "configured_gpu_layers": settings.LLM_NUM_GPU,
        "nvidia_smi_status": gpu_msg,
        "has_nvidia_gpu": has_gpu,
        "ollama_reachable": False,
        "ollama_models": [],
        "instructions": [
            "Use 'ollama ps' while querying to see if the model is running on 100% GPU.",
            "Use 'nvidia-smi' to check VRAM usage.",
            "If Ollama is not using GPU, ensure CUDA is installed and Ollama has GPU support on this OS."
        ]
    }
    try:
        url = f"{settings.OLLAMA_BASE_URL}/api/tags"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            status["ollama_reachable"] = True
            status["ollama_models"] = [m.get("name") for m in response.json().get("models", [])]
    except Exception as e:
        status["error"] = str(e)
    return status

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
