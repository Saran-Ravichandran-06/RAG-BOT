from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import uuid
import shutil
from app.core.config import settings
from typing import List

router = APIRouter()

class ChatResponse(BaseModel):
    chat_id: str
    title: str

@router.post("/create", response_model=ChatResponse)
async def create_chat():
    chat_id = str(uuid.uuid4())
    chat_dir = settings.CHATS_DIR / chat_id
    chat_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metadata
    import json
    meta_file = chat_dir / "metadata.json"
    with open(meta_file, "w") as f:
        json.dump({"title": "New Chat"}, f, indent=2)

    return {"chat_id": chat_id, "title": "New Chat"}

@router.get("/list", response_model=List[ChatResponse])
async def list_chats():
    chats = []
    if settings.CHATS_DIR.exists():
        for chat_dir in settings.CHATS_DIR.iterdir():
            if chat_dir.is_dir():
                title = f"Chat {chat_dir.name[:8]}"
                meta_file = chat_dir / "metadata.json"
                if meta_file.exists():
                    import json
                    try:
                        with open(meta_file, "r") as f:
                            meta = json.load(f)
                            title = meta.get("title", title)
                    except Exception:
                        pass
                chats.append({"chat_id": chat_dir.name, "title": title})
    return chats

@router.delete("/{chat_id}")
async def delete_chat(chat_id: str):
    chat_dir = settings.CHATS_DIR / chat_id
    if chat_dir.exists():
        shutil.rmtree(chat_dir)
        return {"status": "success", "message": "Chat deleted"}
    raise HTTPException(status_code=404, detail="Chat not found")

@router.get("/{chat_id}/history")
async def get_history(chat_id: str):
    chat_dir = settings.CHATS_DIR / chat_id
    if not chat_dir.exists():
        raise HTTPException(status_code=404, detail="Chat not found")
        
    history_file = chat_dir / "messages.json"
    if history_file.exists():
        import json
        with open(history_file, "r") as f:
            return json.load(f)
    return [] 
