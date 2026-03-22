from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from app.ingestion.loader import DocumentLoader
from app.chunking.splitter import RecursiveTokenSplitter
from app.embeddings.model import EmbeddingModel
from app.vectorstore.faiss_store import FaissVectorStore

router = APIRouter()

@router.post("/{chat_id}/upload")
async def upload_file(
    chat_id: str,
    file: UploadFile = File(None),
    url: str = Form(None)
):
    try:
        text = ""
        if file:
            if file.filename.endswith(".pdf"):
                text = await DocumentLoader.load_pdf(file)
            elif file.filename.endswith(".txt"):
                text = await DocumentLoader.load_txt(file)
            else:
                raise HTTPException(status_code=400, detail="Unsupported file format")
        elif url:
            text = DocumentLoader.load_url(url)
        else:
            raise HTTPException(status_code=400, detail="No file or URL provided")

        if not text:
            raise HTTPException(status_code=400, detail="No text extracted")

        # Process pipeline
        splitter = RecursiveTokenSplitter()
        chunks = splitter.split_text(text)
        
        if not chunks:
            return {"status": "warning", "message": "No valid chunks generated"}
            
        embedder = EmbeddingModel.get_instance()
        embeddings = embedder.encode(chunks, convert_to_numpy=True)
        
        vector_store = FaissVectorStore(chat_id)
        vector_store.add_documents(embeddings, chunks)
        
        # Persist upload event to history
        import json
        from app.core.config import settings
        
        chat_dir = settings.CHATS_DIR / chat_id
        chat_dir.mkdir(parents=True, exist_ok=True)
        history_file = chat_dir / "messages.json"
        
        history = []
        if history_file.exists():
            with open(history_file, "r") as f:
                history = json.load(f)
                
        # Determine content message
        content_msg = ""
        title = ""
        if file:
            content_msg = f"Uploaded file: {file.filename}"
            title = file.filename
        elif url:
            content_msg = f"Ingested URL: {url}"
            title = url

        # Update metadata.json with the title
        meta_file = chat_dir / "metadata.json"
        metadata = {"title": title}
        if meta_file.exists():
            with open(meta_file, "r") as f:
                try:
                    metadata = json.load(f)
                    metadata["title"] = title
                except json.JSONDecodeError:
                    pass
        with open(meta_file, "w") as f:
            json.dump(metadata, f, indent=2)
            
        # Add User Message (System event really, but attributed to user action)
        history.append({"role": "user", "content": content_msg})
        
        # Add Assistant Confirmation
        history.append({
            "role": "assistant",
            "content": f"Successfully processed content from {file.filename if file else url}. Added {len(chunks)} chunks to knowledge base."
        })
        
        with open(history_file, "w") as f:
            json.dump(history, f, indent=2)

        return {"status": "success", "chunks_added": len(chunks)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
