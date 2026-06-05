from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from app.vectorstore.faiss_store import FaissVectorStore
from app.embeddings.model import EmbeddingModel
from app.rag.generator import RAGGenerator
from app.evaluation.metrics import Evaluator
from app.core.config import settings
import json
import time

router = APIRouter()

class QueryRequest(BaseModel):
    query: str

LOW_CONFIDENCE_ANSWER = "I don't have enough information to answer that question."

def log_metric(label: str, value: str):
    print(f"[{time.strftime('%H:%M:%S')}] {label}: {value}")

@router.post("/{chat_id}/query")
async def query_chat(chat_id: str, request: QueryRequest):
    try:
        t0 = time.time()

        # 1. Embed Query
        print(f"\n[{time.strftime('%H:%M:%S')}] --- NEW QUERY: {request.query[:50]} ---")
        embedder = EmbeddingModel.get_instance()
        query_emb = embedder.encode([request.query], convert_to_numpy=True)
        t1 = time.time()
        log_metric("Embedding Device", EmbeddingModel.get_device())
        log_metric("Query Embedding", f"{t1-t0:.2f}s")

        # 2. Retrieve
        faiss_cache_start = time.time()
        vector_store = FaissVectorStore.get_store(chat_id)
        t_cache = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] Step 2a: FAISS load/cache took {t_cache-faiss_cache_start:.4f}s")

        retrieval_start = time.time()
        results = vector_store.search(query_emb)
        t2 = time.time()
        log_metric("Retrieval", f"{t2-retrieval_start:.2f}s")

        context = [text for text, score in results]
        best_score = max((score for _, score in results), default=0.0)
        print(f"[{time.strftime('%H:%M:%S')}] Retrieval best_score={best_score:.4f}, results={len(results)}")

        if not context:
            print(f"[{time.strftime('%H:%M:%S')}] NO CONTEXT FOUND.")
            def no_context_stream():
                yield f"data: {json.dumps({'token': LOW_CONFIDENCE_ANSWER})}\n\n"
                yield f"data: {json.dumps({'done': True, 'evaluation': {'groundedness_score': 0, 'faithfulness': True, 'hallucination': False, 'confidence': 'Low'}, 'context': []})}\n\n"
            return StreamingResponse(no_context_stream(), media_type="text/event-stream")

        if best_score < settings.RETRIEVAL_CONFIDENCE_THRESHOLD:
            print(
                f"[{time.strftime('%H:%M:%S')}] Retrieval confidence gate skipped LLM "
                f"(best_score={best_score:.4f}, threshold={settings.RETRIEVAL_CONFIDENCE_THRESHOLD:.4f})"
            )
            def low_confidence_stream():
                yield f"data: {json.dumps({'token': LOW_CONFIDENCE_ANSWER})}\n\n"
                yield f"data: {json.dumps({'done': True, 'evaluation': {'groundedness_score': best_score, 'faithfulness': True, 'hallucination': False, 'confidence': 'Low'}, 'context': results})}\n\n"
            return StreamingResponse(low_confidence_stream(), media_type="text/event-stream")

        context, context_metrics = RAGGenerator.optimize_context(results)
        final_prompt, prompt_metrics = RAGGenerator.build_prompt(context, request.query)
        log_metric("Prompt Build", f"{prompt_metrics['prompt_build_time']:.2f}s")
        log_metric("Retrieved Chunks", str(context_metrics["retrieved_chunks"]))
        log_metric("Used Chunks", str(context_metrics["used_chunks"]))
        log_metric("Context Length", f"{prompt_metrics['context_length']} chars")
        log_metric("Prompt Length", f"{prompt_metrics['prompt_length']} chars")

        def generate_and_persist():
            answer_text = ""
            generation_start = time.time()
            first_token_time = None
            response_token_count = 0

            try:
                for chunk in RAGGenerator.generate_prompt_stream(final_prompt):
                    token = chunk.get("response", "")
                    if token:
                        if first_token_time is None:
                            first_token_time = time.time()
                            log_metric("TTFT", f"{first_token_time - generation_start:.2f}s")
                        response_token_count += 1
                        answer_text += token
                        yield f"data: {json.dumps({'token': token})}\n\n"

                    if chunk.get("done") and chunk.get("eval_count") is not None:
                        response_token_count = chunk["eval_count"]
            except Exception as e:
                if first_token_time is None:
                    first_token_time = time.time()
                    log_metric("TTFT", f"{first_token_time - generation_start:.2f}s")
                if "connection" in str(e).lower() or "connect" in str(e).lower():
                    token = f"Error: Failed to connect to Ollama at {settings.OLLAMA_BASE_URL}. Ensure Ollama is running and accessible."
                else:
                    token = f"Error generating response: {str(e)}"
                response_token_count += 1
                answer_text += token
                yield f"data: {json.dumps({'token': token})}\n\n"

            t3 = time.time()
            generation_time = t3 - generation_start
            tokens_per_second = response_token_count / generation_time if generation_time > 0 else 0.0
            if first_token_time is None:
                log_metric("TTFT", "N/A")
            log_metric("Generation Time", f"{generation_time:.2f}s")
            log_metric("Tokens Generated", str(response_token_count))
            log_metric("Tokens/sec", f"{tokens_per_second:.2f}")

            evaluation = {
                "groundedness_score": 0.0,
                "faithfulness": True,
                "hallucination": False,
                "confidence": "Evaluation Disabled"
            }
            if settings.ENABLE_EVALUATION:
                print(f"[{time.strftime('%H:%M:%S')}] Step 4: Starting Evaluation...")
                evaluation = Evaluator.evaluate(answer_text, context)
                t4 = time.time()
                print(f"[{time.strftime('%H:%M:%S')}] Step 4: Evaluation took {t4-t3:.2f}s")
            else:
                t4 = t3
                print(f"[{time.strftime('%H:%M:%S')}] Step 4: Evaluation SKIPPED")

            yield f"data: {json.dumps({'done': True, 'evaluation': evaluation, 'context': results})}\n\n"

            # Persist History
            chat_dir = settings.CHATS_DIR / chat_id
            chat_dir.mkdir(parents=True, exist_ok=True)
            history_file = chat_dir / "messages.json"

            history = []
            if history_file.exists():
                with open(history_file, "r") as f:
                    history = json.load(f)

            history.append({"role": "user", "content": request.query})
            history.append({
                "role": "assistant",
                "content": answer_text,
                "sources": results,
                "evaluation": evaluation
            })

            with open(history_file, "w") as f:
                json.dump(history, f, indent=2)

            print(f"[{time.strftime('%H:%M:%S')}] --- TOTAL TIME: {t4-t0:.2f}s ---\n")

        return StreamingResponse(generate_and_persist(), media_type="text/event-stream")

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{str(e)}\n\n{traceback.format_exc()}")
