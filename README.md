# TranslateGemma MLX Server

This repository exposes the TranslateGemma MLX model through an OpenAI-compatible FastAPI endpoint (`server.py`) and a simple verification script (`test_call.py`). The primary goal is to convert TranslateGemma’s special structured input format into the standard OpenAI Chat Completions contract and expose an API that translation tools can consume directly.

## Requirements

- Python 3.12 (Apple Silicon recommended for MLX)
- Packages: `mlx_lm`, `fastapi`, `uvicorn`, `pydantic`, `openai`, `pyyaml`

Install them via:

```bash
pip install mlx-lm fastapi uvicorn pydantic openai
```

## server.py

- Loads TranslateGemma weights on demand per request based on the requested model name (from `config.yaml`).
- Unloads the active model when a different model is requested or after a period of inactivity.
- Provides `/health` and `/v1/chat/completions` endpoints that mirror the OpenAI Chat Completions API.
- Provides `/v1/models` and `/v1/models/{model_id}` for model discovery.
- Automatically chunks long texts, detects Chinese vs. non-Chinese content, and uses `mlx_lm` samplers for deterministic or temperature-driven decoding.
- Adds `<end_of_turn>` as an EOS token so generations stop before returning that marker.
- Select the active model by passing `model` in the request body (required; must match a key in `model_map`).
- Accepts optional `src_language_code` / `dst_language_code` request fields (omit both to rely on config defaults) or inline directives like `[zh]->[en]` to override language detection.
- Honors the `TRANSLATEGEMMA_VERBOSE` environment variable so you can enable/disable chunk debug logs and `mlx_lm.generate(..., verbose=True)` without changing API requests.
- Reads configuration defaults from `config.yaml` via `config.py` (PyYAML + Pydantic); environment variables can override individual values.
- Supports `stream=true` with token streaming via `mlx_lm.stream_generate`.

### Configuration

| Env Var | Default | Description |
| --- | --- | --- |
| `TRANSLATEGEMMA_MODEL_IDLE_TIMEOUT` | `600` | Seconds of idle time before the loaded model is unloaded. |
| `MAX_CONTEXT_TOKENS` | `2048` | Total context budget (prompt + output). |
| `DEFAULT_MAX_OUTPUT_TOKENS` | `512` | Fallback `max_tokens` value when API clients omit it. |
| `SRC_LANGUAGE_CODE` | unset | Optional explicit source language code used when no higher-priority input is provided. |
| `DST_LANGUAGE_CODE` | unset | Optional explicit target language code used when no higher-priority input is provided. |
| `TRANSLATEGEMMA_VERBOSE` | `0` | Set to `1`, `true`, or `yes` to print debug logs and run `mlx_lm.generate` in verbose mode. |
| `TRANSLATEGEMMA_CONFIG_FILE` | `config.yaml` | Path to the YAML file that seeds default configuration values. |

Example launch:

```bash
TRANSLATEGEMMA_CONFIG_FILE=config.local.yaml \
SRC_LANGUAGE_CODE=zh \
DST_LANGUAGE_CODE=en \
uvicorn server:app --host 127.0.0.1 --port 8088
```

### Requesting explicit directions

- Inline directive `[xx]->[yy]` (or the single-bracket form `[xx->yy]`) inside the user message has the highest priority; language labels may include spaces and ignore a trailing `language` suffix (e.g., `Japanese Language` == `Japanese`).
- Request fields come next: set both `src_language_code` / `dst_language_code` to a concrete pair (e.g., `"en"` + `"zh"`) to force that direction. If you only set one, the request is rejected.
- You can also prefix the user content with an inline directive such as `[zh]->[en]` followed by a space; the directive will be stripped and the specified source/target pair will be honored.
- If neither directive nor request fields resolve the direction, the server falls back to `config.yaml` when `SRC_LANGUAGE_CODE`/`DST_LANGUAGE_CODE` are set; otherwise it uses CJK-based detection to pick between `zh` and `en`.
- To trigger CJK detection, leave both language fields unset (or set them to `auto`); the server will choose between `zh` and `en`.

### Debug logging

- Set the `TRANSLATEGEMMA_VERBOSE` environment variable when starting the server (e.g., `TRANSLATEGEMMA_VERBOSE=1 bash start_4b.sh`) to stream chunk diagnostics to stdout. Non-streaming requests also pass `verbose=True` into `mlx_lm.generate`; streaming uses `mlx_lm.stream_generate` and does not emit verbose token logs.

### Configuration file

- Edit `config.yaml` to change defaults such as the model map, context limits, or default languages.
- Use `model_map` to map user-facing model names to local MLX paths.
- Use `model_idle_timeout_seconds` to control when inactive models are unloaded.
- Use `lang_alias` in `config.yaml` to map a canonical language code to a list of human-readable aliases (for example `zh: ["simplified chinese", "chinese"]`).
- Use `TRANSLATEGEMMA_CONFIG_FILE=/path/to/custom.yaml` when launching `uvicorn` if you want to point the server at another YAML file.
- Environment variables continue to take precedence over YAML values, so automation scripts and `start_*.sh` remain compatible.

## test_call.py

This script uses the official `openai` Python client to hit the locally running server:

```bash
python test_call.py
```

It assumes the API is served at `http://127.0.0.1:8088/v1` with the dummy key `dummy`. The script submits a plain-text prompt and prints the translated result. Adjust `max_tokens`, `temperature`, or the `messages` payload for your needs.
