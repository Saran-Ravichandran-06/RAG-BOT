import requests
import json
import time
from app.core.config import settings

class OllamaManager:
    @classmethod
    def generate(cls, system: str, prompt: str, temperature: float = None, num_ctx: int = None):
        url = f"{settings.OLLAMA_BASE_URL}/api/generate"
        
        payload = {
            "model": settings.LLM_MODEL,
            "system": system,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature if temperature is not None else settings.LLM_TEMPERATURE,
                "num_ctx": num_ctx if num_ctx is not None else settings.LLM_CONTEXT_LENGTH,
                "num_gpu": settings.LLM_NUM_GPU,
            },
            "keep_alive": settings.LLM_KEEP_ALIVE
        }
        
        print(f"[{time.strftime('%H:%M:%S')}] Ollama Request: model={settings.LLM_MODEL}, layers={settings.LLM_NUM_GPU}")
        
        start = time.time()
        try:
            # Increased timeout to 120s as multi-thousand character prompts take time to process
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            elapsed = time.time() - start
            
            result = response.json()
            print(f"[{time.strftime('%H:%M:%S')}] Ollama Response (tokens: {result.get('eval_count', 'N/A')}) received in {elapsed:.2f}s")
            
            return result
        except Exception as e:
            print(f"Ollama Error: {str(e)}")
            raise e

    @classmethod
    def generate_stream(cls, system: str, prompt: str, temperature: float = None, num_ctx: int = None):
        url = f"{settings.OLLAMA_BASE_URL}/api/generate"
        
        payload = {
            "model": settings.LLM_MODEL,
            "system": system,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": temperature if temperature is not None else settings.LLM_TEMPERATURE,
                "num_ctx": num_ctx if num_ctx is not None else settings.LLM_CONTEXT_LENGTH,
                "num_gpu": settings.LLM_NUM_GPU,
            },
            "keep_alive": settings.LLM_KEEP_ALIVE
        }
        
        print(f"[{time.strftime('%H:%M:%S')}] Ollama Stream Request: model={settings.LLM_MODEL}, layers={settings.LLM_NUM_GPU}")
        
        try:
            response = requests.post(url, json=payload, stream=True, timeout=120)
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    yield json.loads(line)
        except Exception as e:
            print(f"Ollama Stream Error: {str(e)}")
            raise e
