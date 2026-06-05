from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.api import chat, upload, query
from app.core.config import settings
from app.embeddings.model import EmbeddingModel
import requests
import time
import subprocess

def log_startup(label: str, value: str):
    print(f"[{time.strftime('%H:%M:%S')}] {label}: {value}")

def check_nvidia_smi():
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return True, "nvidia-smi is available. GPU detected."
        return False, "nvidia-smi returned non-zero exit code."
    except Exception as e:
        return False, f"nvidia-smi not found or failed: {str(e)}"

def log_torch_gpu_diagnostics():
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        log_startup("torch.cuda.is_available()", str(cuda_available))
        log_startup("PyTorch CUDA version", str(torch.version.cuda))

        if not cuda_available:
            log_startup("Embedding GPU", "CUDA unavailable; embeddings will run on CPU")
            return

        device_index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device_index)
        total_vram_gb = props.total_memory / (1024 ** 3)
        free_vram_gb = None

        if hasattr(torch.cuda, "mem_get_info"):
            free_bytes, _total_bytes = torch.cuda.mem_get_info(device_index)
            free_vram_gb = free_bytes / (1024 ** 3)

        log_startup("GPU name", props.name)
        log_startup("Total VRAM", f"{total_vram_gb:.2f} GB")
        if free_vram_gb is not None:
            log_startup("Available VRAM", f"{free_vram_gb:.2f} GB")
    except Exception as e:
        log_startup("WARNING GPU diagnostics failed", str(e))

def check_ollama_ps_cli():
    try:
        result = subprocess.run(["ollama", "ps"], capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            return None, f"ollama ps returned non-zero exit code: {result.stderr.strip()}"
        return result.stdout.strip(), None
    except Exception as e:
        return None, f"ollama ps unavailable: {str(e)}"

def log_ollama_gpu_diagnostics():
    print(f"[{time.strftime('%H:%M:%S')}] Verifying Ollama GPU usage...")
    has_gpu, gpu_msg = check_nvidia_smi()
    if has_gpu:
        log_startup("GPU Check", gpu_msg)
    else:
        log_startup("WARNING GPU Check", f"{gpu_msg} (Ollama might fallback to CPU)")

    try:
        tags_url = f"{settings.OLLAMA_BASE_URL}/api/tags"
        response = requests.get(tags_url, timeout=5)
        if response.status_code == 200:
            log_startup("Ollama server", "reachable")
            log_startup("Ollama GPU layers configured", str(settings.LLM_NUM_GPU))
        else:
            log_startup("WARNING Ollama server", f"returned status code {response.status_code}")
            return
    except requests.exceptions.RequestException:
        log_startup("WARNING Ollama server", f"could not connect to {settings.OLLAMA_BASE_URL}")
        return

    running_models = []
    try:
        ps_url = f"{settings.OLLAMA_BASE_URL}/api/ps"
        ps_response = requests.get(ps_url, timeout=5)
        if ps_response.status_code == 200:
            running_models = ps_response.json().get("models", [])
            log_startup("Ollama running models", str(len(running_models)))
    except requests.exceptions.RequestException as e:
        log_startup("WARNING Ollama /api/ps", str(e))

    cli_output, cli_error = check_ollama_ps_cli()
    if cli_output:
        processor_lines = [line for line in cli_output.splitlines() if settings.LLM_MODEL in line or "PROCESSOR" in line]
        if processor_lines:
            log_startup("ollama ps", " | ".join(processor_lines))

        output_upper = cli_output.upper()
        if "CPU" in output_upper and "GPU" not in output_upper:
            log_startup("WARNING Ollama GPU", "ollama ps indicates CPU execution")
        elif "GPU" in output_upper:
            log_startup("Ollama GPU", "ollama ps reports GPU execution")
        elif running_models:
            log_startup("WARNING Ollama GPU", "running model found, but processor could not be verified")
    elif cli_error:
        log_startup("WARNING ollama ps", cli_error)

    if not running_models:
        log_startup("Ollama GPU verification", "no active model yet; run a query, then validate with ollama ps and nvidia-smi")

    # Manual validation:
    # - Run `ollama ps` during a query and check the PROCESSOR column for GPU usage.
    # - Run `nvidia-smi` during generation and confirm VRAM/utilization increases.

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Log PyTorch/CUDA diagnostics before loading embeddings
    log_torch_gpu_diagnostics()

    # 2. Preload the embedding model
    print(f"[{time.strftime('%H:%M:%S')}] Preloading Embedding Model ({settings.EMBEDDING_MODEL})...")
    EmbeddingModel.get_instance()
    log_startup("Embedding device", EmbeddingModel.get_device())

    # 3. Verify Ollama GPU visibility and provide validation hints
    log_ollama_gpu_diagnostics()

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
