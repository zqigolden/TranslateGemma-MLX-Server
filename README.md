# TranslateGemma MLX Server

This repository exposes the TranslateGemma MLX model through an OpenAI-compatible FastAPI endpoint (`server.py`) and a simple verification script (`test_call.py`). The primary goal is to convert TranslateGemma’s special structured input format into the standard OpenAI Chat Completions contract and expose an API that translation tools can consume directly.

## Requirements

- Python 3.12 (Apple Silicon recommended for MLX)
- Packages: `mlx_lm`, `fastapi`, `uvicorn`, `pydantic`, `openai`

Install them via:

```bash
pip install mlx-lm fastapi uvicorn pydantic openai
```

## server.py

- Loads the TranslateGemma weights into MLX once at startup (`TRANSLATEGEMMA_MLX_MODEL` env var points to the model directory).
- Provides `/health` and `/v1/chat/completions` endpoints that mirror the OpenAI Chat Completions API.
- Automatically chunks long texts, detects Chinese vs. non-Chinese content, and uses `mlx_lm` samplers for deterministic or temperature-driven decoding.
- Adds `<end_of_turn>` as an EOS token so generations stop before returning that marker.

### Configuration

| Env Var | Default | Description |
| --- | --- | --- |
| `TRANSLATEGEMMA_MLX_MODEL` | `mlx-community/translategemma-4b-it-4bit` | Path to the MLX TranslateGemma model. |
| `MAX_CONTEXT_TOKENS` | `2048` | Total context budget (prompt + output). |
| `DEFAULT_MAX_OUTPUT_TOKENS` | `512` | Fallback `max_tokens` value when API clients omit it. |
| `PRIMARY_LANG_CODE` | `zh` | Source language when the text is detected as CJK-heavy; also the default target when detection cannot decide. |
| `SECONDARY_LANG_CODE` | `en` | Target language for CJK-heavy text; becomes the source language otherwise. |

Example launch:

```bash
TRANSLATEGEMMA_MLX_MODEL=mlx-community/translategemma-4b-it-4bit \
PRIMARY_LANG_CODE=zh \
SECONDARY_LANG_CODE=en \
uvicorn server:app --host 127.0.0.1 --port 8088
```

## test_call.py

This script uses the official `openai` Python client to hit the locally running server:

```bash
python test_call.py
```

It assumes the API is served at `http://127.0.0.1:8088/v1` with the dummy key `dummy`. The script submits a plain-text prompt and prints the translated result. Adjust `max_tokens`, `temperature`, or the `messages` payload for your needs.
