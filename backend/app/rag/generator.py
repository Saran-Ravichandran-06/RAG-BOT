from app.core.config import settings
from app.core.llm import OllamaManager
import re
import time

class RAGGenerator:
    CHUNK_CHAR_LIMIT = 1000
    MAX_CONTEXT_CHARS = 4000

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

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join(text.lower().split())

    @staticmethod
    def _word_set(text: str) -> set[str]:
        return set(re.findall(r"\w+", text.lower()))

    @classmethod
    def _remove_duplicate_sections(cls, text: str) -> str:
        sections = re.split(r"(?<=[.!?])\s+|\n+", text)
        seen = set()
        unique_sections = []

        for section in sections:
            cleaned = section.strip()
            if not cleaned:
                continue

            normalized = cls._normalize_text(cleaned)
            if normalized in seen:
                continue

            seen.add(normalized)
            unique_sections.append(cleaned)

        return " ".join(unique_sections)

    @classmethod
    def _is_redundant(cls, candidate: str, selected: list[str]) -> bool:
        candidate_norm = cls._normalize_text(candidate)
        candidate_words = cls._word_set(candidate)

        if not candidate_words:
            return True

        for existing in selected:
            existing_norm = cls._normalize_text(existing)
            if candidate_norm in existing_norm or existing_norm in candidate_norm:
                return True

            existing_words = cls._word_set(existing)
            if not existing_words:
                continue

            overlap = len(candidate_words & existing_words) / max(1, len(candidate_words | existing_words))
            if overlap >= 0.8:
                return True

        return False

    @classmethod
    def optimize_context(cls, results: list[tuple[str, float]]) -> tuple[list[str], dict]:
        sorted_results = sorted(results, key=lambda item: item[1], reverse=True)
        top_score = sorted_results[0][1] if sorted_results else 0.0
        second_score = sorted_results[1][1] if len(sorted_results) > 1 else 0.0

        if top_score >= 0.8 and top_score - second_score >= 0.12:
            max_chunks = 1
        elif top_score >= 0.65:
            max_chunks = 2
        else:
            max_chunks = 3

        selected = []
        for text, _score in sorted_results:
            cleaned = cls._remove_duplicate_sections(text)[:cls.CHUNK_CHAR_LIMIT]
            if not cleaned or cls._is_redundant(cleaned, selected):
                continue

            selected.append(cleaned)
            if len(selected) >= max_chunks:
                break

        if not selected and sorted_results:
            selected.append(cls._remove_duplicate_sections(sorted_results[0][0])[:cls.CHUNK_CHAR_LIMIT])

        return selected, {
            "retrieved_chunks": len(results),
            "used_chunks": len(selected),
            "top_score": top_score,
        }

    @classmethod
    def build_prompt(cls, context: list[str], question: str) -> tuple[str, dict]:
        prompt_start = time.time()
        context_block = "\n---\n".join(context)

        if len(context_block) > cls.MAX_CONTEXT_CHARS:
            context_block = context_block[:cls.MAX_CONTEXT_CHARS] + "... (truncated)"

        final_prompt = cls.construct_prompt(context_block, question)

        return final_prompt, {
            "prompt_build_time": time.time() - prompt_start,
            "context_length": len(context_block),
            "prompt_length": len(final_prompt),
        }

    @classmethod
    def generate(cls, context: list[str], question: str):
        final_prompt, prompt_metrics = cls.build_prompt(context, question)
        print(
            f"[{time.strftime('%H:%M:%S')}] RAG prompt build: "
            f"{prompt_metrics['prompt_build_time']:.4f}s, length={prompt_metrics['prompt_length']} chars"
        )

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

    @classmethod
    def generate_prompt_stream(cls, prompt: str):
        return OllamaManager.generate_stream(
            system=cls.SYSTEM_PROMPT,
            prompt=prompt
        )

    @classmethod
    def generate_stream(cls, context: list[str], question: str):
        final_prompt, prompt_metrics = cls.build_prompt(context, question)
        print(
            f"[{time.strftime('%H:%M:%S')}] RAG prompt build: "
            f"{prompt_metrics['prompt_build_time']:.4f}s, length={prompt_metrics['prompt_length']} chars"
        )

        try:
            for chunk in cls.generate_prompt_stream(final_prompt):
                yield chunk.get('response', '')
        except Exception as e:
            if "connection" in str(e).lower() or "connect" in str(e).lower():
                yield f"Error: Failed to connect to Ollama at {settings.OLLAMA_BASE_URL}. Ensure Ollama is running and accessible."
            else:
                yield f"Error generating response: {str(e)}"
