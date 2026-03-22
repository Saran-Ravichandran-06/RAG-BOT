from app.core.config import settings
from app.core.llm import OllamaManager

class RAGGenerator:
    SYSTEM_PROMPT = """You are a precise RAG assistant.
Instructions:
1. Answer the question ONLY using the provided Context.
2. If the answer is not in the Context, say: 'I don't have enough information.'
3. Do not use any prior knowledge.
4. Keep the answer concise and direct."""

    @staticmethod
    def construct_prompt(context: str, question: str) -> str:
        return f"""Context:
{context}

Question:
{question}
"""

    @classmethod
    def generate(cls, context: list[str], question: str):
        import time
        # Limit context to avoid 10k+ character prompts that slow down Phi3
        trimmed_context = [c[:1000] for c in context] # limit each chunk
        context_block = "\n---\n".join(trimmed_context)
        
        # Final safety truncation
        if len(context_block) > 4000:
            context_block = context_block[:4000] + "... (truncated)"
            
        final_prompt = cls.construct_prompt(context_block, question)
        
        print(f"[{time.strftime('%H:%M:%S')}] RAG Prompt Length: {len(final_prompt)} chars")
        
        try:
            start_gen = time.time()
            response = OllamaManager.generate(
                system=cls.SYSTEM_PROMPT,
                prompt=final_prompt
            )
            duration = time.time() - start_gen
            print(f"[{time.strftime('%H:%M:%S')}] RAG Generation returned in {duration:.2f}s")
            
            return response['response']
        except Exception as e:
            # Check for connection errors
            if "connection" in str(e).lower() or "connect" in str(e).lower():
                return f"Error: Failed to connect to Ollama at {settings.OLLAMA_BASE_URL}. Ensure Ollama is running and accessible."
            return f"Error generating response: {str(e)}"
