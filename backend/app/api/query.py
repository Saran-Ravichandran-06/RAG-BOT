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

@router.post("/{chat_id}/query")
async def query_chat(chat_id: str, request: QueryRequest):
    try:
        t0 = time.time()

        # 1. Embed Query
        print(f"\n[{time.strftime('%H:%M:%S')}] --- NEW QUERY: {request.query[:50]} ---")
        embedder = EmbeddingModel.get_instance()
        query_emb = embedder.encode([request.query], convert_to_numpy=True)
        t1 = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] Step 1: Embedding took {t1-t0:.2f}s")

        # 2. Retrieve
        faiss_cache_start = time.time()
        vector_store = FaissVectorStore.get_store(chat_id)
        t_cache = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] Step 2a: FAISS load/cache took {t_cache-faiss_cache_start:.4f}s")

        retrieval_start = time.time()
        results = vector_store.search(query_emb)
        t2 = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] Step 2b: Retrieval took {t2-retrieval_start:.4f}s")

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

        print(f"[{time.strftime('%H:%M:%S')}] Step 3: Starting generation...")

        def generate_and_persist():
            answer_text = ""
            gen_start = time.time()
            first_token_time = None
            for token in RAGGenerator.generate_stream(context, request.query):
                if first_token_time is None and token:
                    first_token_time = time.time()
                    print(f"[{time.strftime('%H:%M:%S')}] Step 3a: TTFT took {first_token_time-gen_start:.2f}s")
                answer_text += token
                yield f"data: {json.dumps({'token': token})}\n\n"

            t3 = time.time()
            if first_token_time is None:
                print(f"[{time.strftime('%H:%M:%S')}] Step 3a: TTFT unavailable; no token received")
            print(f"[{time.strftime('%H:%M:%S')}] Step 3b: Total generation took {t3-gen_start:.2f}s")

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
