from app.core.config import settings

class RecursiveTokenSplitter:
    def __init__(self, chunk_size: int = settings.CHUNK_SIZE, overlap: int = settings.CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split_text(self, text: str) -> list[str]:
        # Simple character-based splitting approximated for speed
        # For production with varying tokenizers, one might use tiktoken or sentence-piece
        # Here we use a safe approximation: 1 token ~ 4 chars
        # But since we're using sentence-transformers, it has its own limit.
        # We'll split by sentences first, then group them.
        
        # A simple naive approach to start, can be improved to use NLTK or Spacy if strictly required
        # but standard string manipulation is often sufficient for basic RAG.
        
        # NOTE: This is a simplified recursive splitter. 
        # In a real heavy production setup, we might use LangChain's splitter logic re-implemented.
        
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            current_len_word = len(word) + 1 # +1 for space
            if current_length + current_len_word > self.chunk_size * 4: # approx 4 chars per token
                chunks.append(" ".join(current_chunk))
                
                # Handle overlap
                # Keep last N words that fit in overlap
                overlap_len = 0
                overlap_chunk = []
                for w in reversed(current_chunk):
                    if overlap_len + len(w) + 1 < self.overlap * 4:
                        overlap_chunk.insert(0, w)
                        overlap_len += len(w) + 1
                    else:
                        break
                
                current_chunk = overlap_chunk
                current_length = overlap_len
            
            current_chunk.append(word)
            current_length += len(word) + 1
            
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks
