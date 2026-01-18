from __future__ import annotations

import json
import re
import time
import uuid
import asyncio
from typing import Any, Dict, List, Literal, Optional, Tuple

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
from config import AUTO_DETECT_SENTINEL, build_lang_label_map, load_config


# ========= Configuration =========
CONFIG = load_config()
MODEL_PATH = CONFIG.model_path
MAX_CONTEXT_TOKENS = CONFIG.max_context_tokens
DEFAULT_MAX_OUTPUT_TOKENS = CONFIG.default_max_output_tokens
MIN_INPUT_BUDGET = CONFIG.min_input_budget  # Prevent the input budget from becoming too small in edge cases
EXTRA_STOP_STRINGS = CONFIG.extra_stop_strings
SRC_LANGUAGE_CODE = CONFIG.src_language_code
DST_LANGUAGE_CODE = CONFIG.dst_language_code
LANG_LABEL_MAP = build_lang_label_map(CONFIG.lang_alias)
VERBOSE_LOGGING = CONFIG.verbose_logging
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
    src_language_code: Optional[str] = None
    dst_language_code: Optional[str] = None


# ========= Language direction detection (default zh->en, otherwise en->zh) =========
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_LANG_DIRECTIVE_RE = re.compile(
    r"^\[(?P<src>[^\]]+?)\]\s*->\s*\[(?P<tgt>[^\]]+?)\]\s*",
)
_LANG_DIRECTIVE_SINGLE_RE = re.compile(
    r"^\[(?P<inner>[^\]]+?)\]\s*",
)

def _normalize_lang_code(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    raw = " ".join(value.strip().split())
    if not raw:
        return None
    lowered = raw.lower()
    alias_key = _strip_language_suffix(lowered)
    if not alias_key or alias_key == AUTO_DETECT_SENTINEL:
        return None
    mapped = LANG_LABEL_MAP.get(alias_key)
    return mapped if mapped is not None else raw


def _strip_language_suffix(value: str) -> str:
    suffix = " language"
    if value.endswith(suffix):
        return value[: -len(suffix)].rstrip()
    return value

def _detect_direction(text: str, lang_pair: Tuple[str, str]) -> Tuple[str, str]:
    src_lang, dst_lang = lang_pair
    if not text.strip():
        return src_lang, dst_lang
    cjk_count = len(_CJK_RE.findall(text))
    total = len([ch for ch in text if not ch.isspace()])
    ratio = (cjk_count / total) if total > 0 else 0.0
    if cjk_count >= 10 and ratio >= 0.30:
        return (src_lang, dst_lang)
    return (dst_lang, src_lang)

def extract_lang_directive(text: str) -> Tuple[Optional[str], Optional[str], str]:
    match = _LANG_DIRECTIVE_RE.match(text)
    if not match:
        match = _LANG_DIRECTIVE_SINGLE_RE.match(text)
        if not match:
            return None, None, text
        inner = match.group("inner")
        if "->" not in inner:
            return None, None, text
        src_raw, tgt_raw = inner.split("->", 1)
        src = src_raw.strip()
        tgt = tgt_raw.strip()
        if not src or not tgt:
            return None, None, text
        stripped_text = text[match.end():].lstrip()
        return src, tgt, stripped_text
    stripped_text = text[match.end():].lstrip()
    return match.group("src").strip(), match.group("tgt").strip(), stripped_text

def _resolve_lang_pair(src_value: Optional[str], dst_value: Optional[str]) -> Optional[Tuple[str, str]]:
    norm_src = _normalize_lang_code(src_value)
    norm_dst = _normalize_lang_code(dst_value)

    if not norm_src and not norm_dst:
        return None

    if not norm_src or not norm_dst:
        raise HTTPException(
            status_code=400,
            detail="Specify both src_language_code and dst_language_code when overriding language direction.",
        )

    return norm_src, norm_dst


def decide_target_lang(text: str, requested_src: Optional[str], requested_dst: Optional[str]) -> Tuple[str, str]:
    # Request-level resolution
    request_pair = _resolve_lang_pair(requested_src, requested_dst)
    if request_pair:
        return request_pair

    # Configuration-level fallback (lowest priority)
    config_pair = _resolve_lang_pair(SRC_LANGUAGE_CODE, DST_LANGUAGE_CODE)
    if config_pair:
        return config_pair

    return _detect_direction(text, ("zh", "en"))


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
        verbose=VERBOSE_LOGGING,
    ).strip()

    completion_tokens = tok_len(out)
    return out, prompt_tokens, completion_tokens


def _split_stream_text(text: str, chunk_size: int) -> List[str]:
    if not text:
        return []
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


def _format_stream_event(base: Dict[str, Any], delta: Dict[str, str], finish_reason: Optional[str]) -> str:
    payload = dict(base)
    payload["choices"] = [{
        "index": 0,
        "delta": delta,
        "finish_reason": finish_reason,
    }]
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.get("/health")
def health():
    return {
        "ok": True,
        "model_path": MODEL_PATH,
        "max_context_tokens": MAX_CONTEXT_TOKENS,
    }


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionsRequest, request: Request):
    if VERBOSE_LOGGING:
        try:
            raw_body = await request.body()
            raw_text = raw_body.decode("utf-8", errors="replace")
            print(f"[TranslateGemma][DEBUG] raw_request={raw_text}", flush=True)
        except Exception as exc:
            print(f"[TranslateGemma][DEBUG] raw_request_error={exc!r}", flush=True)
        print(f"[TranslateGemma][DEBUG] parsed_request={req.dict()}", flush=True)

    user_texts = [m.content for m in req.messages if m.role == "user"]
    if not user_texts:
        raise HTTPException(status_code=400, detail="No user message provided")

    raw_text = user_texts[-1]
    directive_src, directive_dst, stripped_text = extract_lang_directive(raw_text)
    full_text = stripped_text.strip()
    if not full_text:
        raise HTTPException(status_code=400, detail="Empty input text")

    requested_src = directive_src or req.src_language_code
    requested_dst = directive_dst or req.dst_language_code
    src, tgt = decide_target_lang(full_text, requested_src, requested_dst)
    if VERBOSE_LOGGING:
        print(
            f"[TranslateGemma][DEBUG] input_src={src} input_dst={tgt} raw={raw_text!r}",
            flush=True,
        )
        if raw_text != full_text:
            print(
                f"[TranslateGemma][DEBUG] normalized_text={full_text!r}",
                flush=True,
            )
    max_output_tokens = req.max_tokens if req.max_tokens is not None else DEFAULT_MAX_OUTPUT_TOKENS
    if max_output_tokens <= 0:
        max_output_tokens = DEFAULT_MAX_OUTPUT_TOKENS
    if max_output_tokens >= MAX_CONTEXT_TOKENS:
        max_output_tokens = min(512, MAX_CONTEXT_TOKENS // 4)

    chunks = chunk_text_to_fit(full_text, src, tgt, max_output_tokens=max_output_tokens)
    if VERBOSE_LOGGING:
        print(
            f"[TranslateGemma][DEBUG] src={src} dst={tgt} chunks={len(chunks)} tokens={max_output_tokens}",
            flush=True,
        )

    # ====== Concurrency lock: serialize generate to avoid overlapping requests ======
    t0 = time.time()
    async with GEN_LOCK:
        outputs: List[str] = []
        prompt_tokens_sum = 0
        completion_tokens_sum = 0

        for idx, ch in enumerate(chunks, start=1):
            if VERBOSE_LOGGING:
                print(
                    f"[TranslateGemma][DEBUG] Translating chunk {idx}/{len(chunks)} chars={len(ch)}",
                    flush=True,
                )
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

    if req.stream:
        response_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        model_name = req.model or MODEL_PATH
        chunk_size = 200

        def event_stream():
            base = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
            }
            yield _format_stream_event(base, {"role": "assistant"}, None)
            for piece in _split_stream_text(translated, chunk_size):
                if piece:
                    yield _format_stream_event(base, {"content": piece}, None)
            yield _format_stream_event(base, {}, "stop")
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

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
