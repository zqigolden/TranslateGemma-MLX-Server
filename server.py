from __future__ import annotations

import os
import re
import time
import uuid
import asyncio
from typing import Any, Dict, List, Literal, Optional, Tuple

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler


# ========= Configuration =========
MODEL_PATH = os.environ.get(
    "TRANSLATEGEMMA_MLX_MODEL",
    "mlx-community/translategemma-4b-it-4bit",
)

MAX_CONTEXT_TOKENS = int(os.environ.get("MAX_CONTEXT_TOKENS", "2048"))
DEFAULT_MAX_OUTPUT_TOKENS = int(os.environ.get("DEFAULT_MAX_OUTPUT_TOKENS", "512"))
MIN_INPUT_BUDGET = 256  # Prevent the input budget from becoming too small in edge cases
EXTRA_STOP_STRINGS = ("<end_of_turn>",)
# Configurable translation direction defaults
PRIMARY_LANG_CODE = os.environ.get("PRIMARY_LANG_CODE", "zh")
SECONDARY_LANG_CODE = os.environ.get("SECONDARY_LANG_CODE", "en")
print(MODEL_PATH)
model, tokenizer = load(
    MODEL_PATH,
)
for stop in EXTRA_STOP_STRINGS:
    try:
        tokenizer.add_eos_token(stop)
    except ValueError:
        # Some tokenizers may not expose that special token; ignore the error
        pass

app = FastAPI(title="TranslateGemma MLX OpenAI-Compatible Service", version="1.1.0")

# ========= Concurrency lock: at most one generate call at a time =========
GEN_LOCK = asyncio.Lock()


# ========= OpenAI compatible schema =========
class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionsRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    temperature: float = 0.0
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False


# ========= Language direction detection (default zh->en, otherwise en->zh) =========
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")

def decide_target_lang(text: str) -> Tuple[str, str]:
    if not text.strip():
        return ("auto", PRIMARY_LANG_CODE)
    cjk_count = len(_CJK_RE.findall(text))
    total = len([ch for ch in text if not ch.isspace()])
    ratio = (cjk_count / total) if total > 0 else 0.0
    if cjk_count >= 10 and ratio >= 0.30:
        return (PRIMARY_LANG_CODE, SECONDARY_LANG_CODE)
    return (SECONDARY_LANG_CODE, PRIMARY_LANG_CODE)


# ========= Token counting =========
def tok_len(s: str) -> int:
    return len(tokenizer.encode(s))


# ========= Chunking =========
def split_by_paragraphs(text: str) -> List[str]:
    parts = re.split(r"\n{2,}", text.strip())
    return [p.strip() for p in parts if p.strip()]

def split_by_sentences(text: str) -> List[str]:
    s = text.strip()
    if not s:
        return []
    chunks = re.split(r"(?<=[。！？!?\.])\s+", s)
    return [c.strip() for c in chunks if c.strip()]

def make_messages_structured(chunk_text: str, src: str, tgt: str) -> List[Dict[str, Any]]:
    # Maintain the validated structured content: [{source_lang_code, target_lang_code, text}]
    content = [{
        "type": "text",
        "source_lang_code": src,
        "target_lang_code": tgt,
        "text": chunk_text,
    }]
    return [{"role": "user", "content": content}]

def build_prompt(chunk_text: str, src: str, tgt: str) -> str:
    messages = make_messages_structured(chunk_text, src, tgt)
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    # Fallback (rarely hit)
    return f"Translate from {src} to {tgt}. Return only the translation.\n\n{chunk_text}"

def chunk_text_to_fit(text: str, src: str, tgt: str, max_output_tokens: int) -> List[str]:
    empty_prompt = build_prompt("", src, tgt)
    overhead = tok_len(empty_prompt)
    budget = MAX_CONTEXT_TOKENS - max_output_tokens - overhead
    if budget < MIN_INPUT_BUDGET:
        budget = MIN_INPUT_BUDGET

    paras = split_by_paragraphs(text)
    chunks: List[str] = []
    cur = ""

    def flush():
        nonlocal cur
        if cur.strip():
            chunks.append(cur.strip())
        cur = ""

    for p in paras:
        cand = (cur + "\n\n" + p).strip() if cur else p
        if tok_len(cand) <= budget:
            cur = cand
            continue

        if cur:
            flush()

        if tok_len(p) > budget:
            sents = split_by_sentences(p)
            buf = ""
            for s in sents:
                cand2 = (buf + " " + s).strip() if buf else s
                if tok_len(cand2) <= budget:
                    buf = cand2
                else:
                    if buf:
                        chunks.append(buf)
                        buf = s
                    else:
                        # Sentence still too long: binary search the longest prefix within budget
                        long = s
                        while long:
                            lo, hi = 0, len(long)
                            best = 0
                            while lo <= hi:
                                mid = (lo + hi) // 2
                                piece = long[:mid]
                                if tok_len(piece) <= budget:
                                    best = mid
                                    lo = mid + 1
                                else:
                                    hi = mid - 1
                            piece = long[:best].strip()
                            if piece:
                                chunks.append(piece)
                            long = long[best:].strip()
                        buf = ""
            if buf:
                chunks.append(buf)
        else:
            cur = p

    flush()
    return chunks


# ========= Translate a chunk + usage accounting =========
def translate_chunk(chunk_text: str, src: str, tgt: str, max_output_tokens: int, temperature: float) -> Tuple[str, int, int]:
    prompt = build_prompt(chunk_text, src, tgt)
    prompt_tokens = tok_len(prompt)

    sampler = make_sampler(temperature)

    out = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_output_tokens,
        sampler=sampler,
        verbose=False,
    ).strip()

    completion_tokens = tok_len(out)
    return out, prompt_tokens, completion_tokens


@app.get("/health")
def health():
    return {
        "ok": True,
        "model_path": MODEL_PATH,
        "max_context_tokens": MAX_CONTEXT_TOKENS,
    }


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionsRequest):
    if req.stream:
        raise HTTPException(status_code=400, detail="stream=true not supported in this server")

    user_texts = [m.content for m in req.messages if m.role == "user"]
    if not user_texts:
        raise HTTPException(status_code=400, detail="No user message provided")

    full_text = user_texts[-1].strip()
    if not full_text:
        raise HTTPException(status_code=400, detail="Empty input text")

    src, tgt = decide_target_lang(full_text)
    max_output_tokens = req.max_tokens if req.max_tokens is not None else DEFAULT_MAX_OUTPUT_TOKENS
    if max_output_tokens <= 0:
        max_output_tokens = DEFAULT_MAX_OUTPUT_TOKENS
    if max_output_tokens >= MAX_CONTEXT_TOKENS:
        max_output_tokens = min(512, MAX_CONTEXT_TOKENS // 4)

    chunks = chunk_text_to_fit(full_text, src, tgt, max_output_tokens=max_output_tokens)

    # ====== Concurrency lock: serialize generate to avoid overlapping requests ======
    t0 = time.time()
    async with GEN_LOCK:
        outputs: List[str] = []
        prompt_tokens_sum = 0
        completion_tokens_sum = 0

        for ch in chunks:
            out, pt, ct = translate_chunk(
                ch, src, tgt,
                max_output_tokens=max_output_tokens,
                temperature=req.temperature,
            )
            outputs.append(out)
            prompt_tokens_sum += pt
            completion_tokens_sum += ct

    translated = "\n\n".join(outputs).strip()
    latency_ms = int((time.time() - t0) * 1000)

    # OpenAI compatible response + estimated usage
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model or MODEL_PATH,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": translated},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens_sum,
            "completion_tokens": completion_tokens_sum,
            "total_tokens": prompt_tokens_sum + completion_tokens_sum,
        },
        "_meta": {
            "src_lang": src,
            "tgt_lang": tgt,
            "chunks": len(chunks),
            "latency_ms": latency_ms,
            "max_context_tokens": MAX_CONTEXT_TOKENS,
            "max_output_tokens": max_output_tokens,
        },
    }
