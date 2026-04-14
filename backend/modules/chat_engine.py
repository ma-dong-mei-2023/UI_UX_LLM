"""
Chat engine for instruction-following (Chapter 7).
"""
from __future__ import annotations
import sys
import torch
from config import PKG_DIR

if str(PKG_DIR) not in sys.path:
    sys.path.insert(0, str(PKG_DIR))

from llms_from_scratch.ch05 import generate, text_to_token_ids, token_ids_to_text
from llms_from_scratch.ch07 import format_input


def generate_response(
    model,
    tokenizer,
    instruction: str,
    input_text: str = "",
    temperature: float = 0.7,
    top_k: int = 50,
    max_new_tokens: int = 256,
    context_size: int = 1024,
    device: torch.device = None,
) -> str:
    """Generate a response to an instruction."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    prompt = format_input({"instruction": instruction, "input": input_text})
    prompt += "\n\n### Response:\n"

    model.eval()
    with torch.no_grad():
        idx = text_to_token_ids(prompt, tokenizer).to(device)
        output_ids = generate(
            model=model,
            idx=idx,
            max_new_tokens=max_new_tokens,
            context_size=context_size,
            temperature=temperature,
            top_k=top_k,
            eos_id=50256,  # <|endoftext|>
        )

    full_text = token_ids_to_text(output_ids, tokenizer)
    # Extract only the response part
    response_marker = "### Response:\n"
    if response_marker in full_text:
        response = full_text.split(response_marker)[-1].strip()
    else:
        response = full_text[len(prompt):].strip()

    return response


def format_instruction_preview(instruction: str, input_text: str = "") -> str:
    """Return the formatted instruction prompt for preview."""
    return format_input({"instruction": instruction, "input": input_text}) + "\n\n### Response:\n"
