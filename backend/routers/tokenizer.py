from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import tiktoken
from modules.bpe_tokenizer import BPETokenizerSimple

router = APIRouter(prefix="/api/tokenizer", tags=["tokenizer"])

# Global BPE tokenizer instance (trained per request for small vocab)
_bpe_instance = BPETokenizerSimple()
_bpe_trained = False


class TrainRequest(BaseModel):
    text: str
    vocab_size: int = 500


class EncodeRequest(BaseModel):
    text: str


class CompareRequest(BaseModel):
    text: str
    bpe_vocab_size: int = 500


@router.post("/bpe/train")
def train_bpe(req: TrainRequest):
    """Train BPE and return all merge steps."""
    global _bpe_instance, _bpe_trained
    if len(req.text) < 10:
        raise HTTPException(400, "Text too short (minimum 10 characters)")
    if req.vocab_size < 260 or req.vocab_size > 5000:
        raise HTTPException(400, "vocab_size must be between 260 and 5000")

    _bpe_instance = BPETokenizerSimple()
    steps = list(_bpe_instance.train_step_by_step(req.text, req.vocab_size))
    _bpe_trained = True

    return {
        "steps": steps,
        "total_merges": len(steps),
        "final_vocab_size": len(_bpe_instance.vocab),
        "vocab_sample": _bpe_instance.get_vocab_list(50),
    }


@router.post("/bpe/encode")
def encode_bpe(req: EncodeRequest):
    if not _bpe_trained:
        raise HTTPException(400, "BPE tokenizer not trained yet. Train first.")
    token_ids = _bpe_instance.encode(req.text)
    tokens = [_bpe_instance._decode_token(t) for t in token_ids]
    return {
        "token_ids": token_ids,
        "tokens": tokens,
        "num_tokens": len(token_ids),
    }


@router.post("/bpe/decode")
def decode_bpe(req: EncodeRequest):
    # req.text contains space-separated token IDs
    try:
        ids = [int(x) for x in req.text.split()]
        return {"text": _bpe_instance.decode(ids)}
    except ValueError:
        raise HTTPException(400, "Invalid token IDs")


@router.post("/tiktoken/encode")
def encode_tiktoken(req: EncodeRequest):
    enc = tiktoken.get_encoding("gpt2")
    token_ids = enc.encode(req.text)
    tokens = [enc.decode([t]) for t in token_ids]
    return {
        "token_ids": token_ids,
        "tokens": tokens,
        "num_tokens": len(token_ids),
    }


@router.post("/tiktoken/decode")
def decode_tiktoken(req: EncodeRequest):
    try:
        enc = tiktoken.get_encoding("gpt2")
        ids = [int(x) for x in req.text.split()]
        return {"text": enc.decode(ids)}
    except Exception as e:
        raise HTTPException(400, str(e))


@router.post("/compare")
def compare_tokenizers(req: CompareRequest):
    """Compare custom BPE vs tiktoken on same text."""
    # tiktoken
    enc = tiktoken.get_encoding("gpt2")
    tiktoken_ids = enc.encode(req.text)
    tiktoken_tokens = [enc.decode([t]) for t in tiktoken_ids]

    # Custom BPE
    bpe = BPETokenizerSimple()
    # Train on input text itself for demo
    list(bpe.train_step_by_step(req.text, req.bpe_vocab_size))
    bpe_ids = bpe.encode(req.text)
    bpe_tokens = [bpe._decode_token(t) for t in bpe_ids]

    return {
        "tiktoken": {
            "token_ids": tiktoken_ids,
            "tokens": tiktoken_tokens,
            "num_tokens": len(tiktoken_ids),
            "vocab_size": enc.n_vocab,
        },
        "custom_bpe": {
            "token_ids": bpe_ids,
            "tokens": bpe_tokens,
            "num_tokens": len(bpe_ids),
            "vocab_size": len(bpe.vocab),
        },
        "compression_tiktoken": round(len(req.text.encode()) / max(len(tiktoken_ids), 1), 2),
        "compression_bpe": round(len(req.text.encode()) / max(len(bpe_ids), 1), 2),
    }
