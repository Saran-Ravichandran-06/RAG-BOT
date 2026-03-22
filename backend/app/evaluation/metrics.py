from sentence_transformers import util
from app.embeddings.model import EmbeddingModel
from app.core.config import settings
from app.core.llm import OllamaManager

class Evaluator:
    @staticmethod
    def check_groundedness(answer: str, context: list[str]) -> float:
        if not context:
            return 0.0
            
        model = EmbeddingModel.get_instance()
        
        # Split answer into significant sentences (simplified)
        answer_sentences = [s.strip() for s in answer.split('.') if len(s.strip()) > 10]
        if not answer_sentences:
            return 0.0
            
        answer_embs = model.encode(answer_sentences, convert_to_tensor=True)
        context_embs = model.encode(context, convert_to_tensor=True)
        
        # Calculate max similarity for each answer sentence against any context chunk
        scores = []
        for ans_emb in answer_embs:
            sims = util.cos_sim(ans_emb, context_embs)
            max_sim = float(sims.max())
            scores.append(max_sim)
            
        if not scores:
            return 0.0
            
        return sum(scores) / len(scores)

    @staticmethod
    def check_faithfulness(answer: str, context: list[str]) -> bool:
        # LLM-as-a-judge
        context_block = "\n".join(context)
        prompt = f"""Context:
{context_block}

Answer:
{answer}

Does the answer introduce facts NOT present in the context?
Answer ONLY "YES" or "NO".
"""
        try:
            # Optimize for evaluation: 0 temperature and small context
            response = OllamaManager.generate(
                system="You are a strict evaluator.",
                prompt=prompt,
                temperature=0.0,
                num_ctx=1024 
            )
            result = response['response'].strip().upper()
            return "YES" not in result
        except:
            return False # Fail safe

    @classmethod
    def evaluate(cls, answer: str, context: list[str]):
        groundedness = cls.check_groundedness(answer, context)
        faithfulness = cls.check_faithfulness(answer, context)
        
        hallucination = False
        if groundedness < settings.HALLUCINATION_THRESHOLD or not faithfulness:
            hallucination = True
            
        return {
            "groundedness_score": groundedness,
            "faithfulness": faithfulness,
            "hallucination": hallucination,
            "confidence": "High" if groundedness > 0.8 else "Low"
        }
