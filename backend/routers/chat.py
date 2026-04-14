from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import json
from config import INSTRUCTION_DATA_PATH
from modules.chat_engine import format_instruction_preview

router = APIRouter(prefix="/api/chat", tags=["chat"])


class MessageRequest(BaseModel):
    instruction: str
    input_text: str = ""
    temperature: float = 0.7
    top_k: int = 50
    max_new_tokens: int = 256


class FormatPreviewRequest(BaseModel):
    instruction: str
    input_text: str = ""


@router.post("/format-preview")
def format_preview(req: FormatPreviewRequest):
    formatted = format_instruction_preview(req.instruction, req.input_text)
    return {"formatted": formatted}


@router.get("/instruction-data/sample")
def get_instruction_sample(limit: int = 20, offset: int = 0):
    if not INSTRUCTION_DATA_PATH.exists():
        raise HTTPException(404, "Instruction data file not found")
    with open(INSTRUCTION_DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    total = len(data)
    sample = data[offset: offset + limit]
    return {"total": total, "offset": offset, "limit": limit, "data": sample}


@router.post("/message")
def send_message(req: MessageRequest):
    """Simple non-streaming chat (for when model is loaded in memory)."""
    # Note: actual model inference requires a loaded model.
    # This endpoint returns the formatted prompt when no model is loaded.
    formatted = format_instruction_preview(req.instruction, req.input_text)
    return {
        "response": "[Model not loaded. Start training or load pretrained weights to enable chat.]",
        "formatted_prompt": formatted,
        "model_loaded": False,
    }
