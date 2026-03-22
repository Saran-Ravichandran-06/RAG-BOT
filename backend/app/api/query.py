from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.vectorstore.faiss_store import FaissVectorStore
from app.embeddings.model import EmbeddingModel
from app.rag.generator import RAGGenerator
from app.evaluation.metrics import Evaluator
from app.core.config import settings

router = APIRouter()

class QueryRequest(BaseModel):
    query: str

@router.post("/{chat_id}/query")
async def query_chat(chat_id: str, request: QueryRequest):
    try:
        import time
        t0 = time.time()
        
        # 1. Embed Query
        print(f"\n[{time.strftime('%H:%M:%S')}] --- NEW QUERY: {request.query[:50]} ---")
        embedder = EmbeddingModel.get_instance()
        query_emb = embedder.encode([request.query], convert_to_numpy=True)
        t1 = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] Step 1: Embedding took {t1-t0:.2f}s")
        
        # 2. Retrieve
        vector_store = FaissVectorStore(chat_id)
        results = vector_store.search(query_emb)
        t2 = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] Step 2: Retrieval took {t2-t1:.2f}s")
        
        context = [text for text, score in results]
        
        if not context:
            print(f"[{time.strftime('%H:%M:%S')}] NO CONTEXT FOUND.")
            return {
                "answer": "I don't have enough information.",
                "context": [],
                "evaluation": {
                    "groundedness_score": 0,
                    "faithfulness": True,
                    "hallucination": False,
                    "confidence": "Low"
                }
            }

        # 3. Generate
        print(f"[{time.strftime('%H:%M:%S')}] Step 3: Starting generation...")
        answer = RAGGenerator.generate(context, request.query)
        if answer.startswith("Error"):
            raise HTTPException(status_code=500, detail=answer)
            
        t3 = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] Step 3: Generation took {t3-t2:.2f}s")
        
        # 4. Evaluate
        evaluation = {
            "groundedness_score": 0.0,
            "faithfulness": True,
            "hallucination": False,
            "confidence": "Evaluation Disabled"
        }
        if settings.ENABLE_EVALUATION:
            print(f"[{time.strftime('%H:%M:%S')}] Step 4: Starting Evaluation...")
            evaluation = Evaluator.evaluate(answer, context)
            t4 = time.time()
            print(f"[{time.strftime('%H:%M:%S')}] Step 4: Evaluation took {t4-t3:.2f}s")
        else:
            t4 = t3
            print(f"[{time.strftime('%H:%M:%S')}] Step 4: Evaluation SKIPPED")
        
        response_data = {
            "answer": answer,
            "context": results, # Return text + score
            "evaluation": evaluation
        }

        # 5. Persist History
        import json
        import os
        
        chat_dir = settings.CHATS_DIR / chat_id
        chat_dir.mkdir(parents=True, exist_ok=True)
        history_file = chat_dir / "messages.json"
        
        history = []
        if history_file.exists():
            with open(history_file, "r") as f:
                history = json.load(f)
        
        # Add User Message
        history.append({"role": "user", "content": request.query})
        
        # Add Assistant Message (simplified for storage)
        history.append({
            "role": "assistant", 
            "content": answer,
            "sources": results,
            "evaluation": evaluation
        })
        
        with open(history_file, "w") as f:
            json.dump(history, f, indent=2)
            
        print(f"[{time.strftime('%H:%M:%S')}] --- TOTAL TIME: {t4-t0:.2f}s ---\n")
        return response_data

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{str(e)}\n\n{traceback.format_exc()}")
