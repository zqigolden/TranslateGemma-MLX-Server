from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from pydantic import BaseModel, Field, validator

AUTO_DETECT_SENTINEL = "auto"
DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")
_TRUE_VALUES = {"1", "true", "yes", "on"}


class TranslateGemmaConfig(BaseModel):
    model_map: Dict[str, str] = Field(default_factory=dict)
    max_context_tokens: int = 2048
    default_max_output_tokens: int = 512
    min_input_budget: int = 256
    extra_stop_strings: Tuple[str, ...] = Field(default_factory=lambda: ("<end_of_turn>",))
    model_idle_timeout_seconds: int = 600
    src_language_code: Optional[str] = None
    dst_language_code: Optional[str] = None
    lang_alias: Dict[str, List[str]] = Field(default_factory=dict)
    verbose_logging: bool = False

    @validator("extra_stop_strings", pre=True)
    def _ensure_tuple(cls, value: Any) -> Tuple[str, ...]:
        if value is None:
            return ("<end_of_turn>",)
        if isinstance(value, (list, tuple)):
            return tuple(str(item) for item in value if str(item))
        return (str(value),)

    @validator("model_map", pre=True)
    def _normalize_model_map(cls, value: Any) -> Dict[str, str]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError("model_map must be a mapping of model names to paths.")
        normalized: Dict[str, str] = {}
        for key, path in value.items():
            name = " ".join(str(key).strip().split())
            if not name:
                continue
            if path is None:
                continue
            path_value = str(path).strip()
            if not path_value:
                continue
            normalized[name] = path_value
        return normalized

    @validator("src_language_code", "dst_language_code", pre=True)
    def _normalize_lang(cls, value: Optional[str]) -> Optional[str]:
        return _normalize_lang_value(value)

    @validator("lang_alias", pre=True)
    def _normalize_lang_alias(cls, value: Any) -> Dict[str, List[str]]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError("lang_alias must be a mapping of language codes to alias lists.")
        normalized: Dict[str, List[str]] = {}
        for key, aliases in value.items():
            norm_key = _normalize_lang_value(key)
            if not norm_key:
                continue
            if aliases is None:
                continue
            if isinstance(aliases, (list, tuple, set)):
                items = aliases
            else:
                items = [aliases]
            seen = set()
            norm_aliases: List[str] = []
            for item in items:
                norm_alias = _normalize_lang_key(item)
                if not norm_alias or norm_alias in seen:
                    continue
                seen.add(norm_alias)
                norm_aliases.append(norm_alias)
            if norm_aliases:
                normalized[norm_key] = norm_aliases
        return normalized


def _normalize_lang_value(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    raw = " ".join(str(value).strip().split())
    if not raw:
        return None
    lowered = raw.lower()
    if lowered == AUTO_DETECT_SENTINEL:
        return None
    return raw


def _normalize_lang_key(value: Any) -> Optional[str]:
    if value is None:
        return None
    cleaned = " ".join(str(value).strip().lower().split())
    cleaned = _strip_language_suffix(cleaned)
    if not cleaned or cleaned == AUTO_DETECT_SENTINEL:
        return None
    return cleaned


def build_lang_label_map(lang_alias: Dict[str, List[str]]) -> Dict[str, str]:
    label_map: Dict[str, str] = {}
    for lang_code, aliases in lang_alias.items():
        for alias in aliases:
            label_map[alias] = lang_code
    return label_map


def _strip_language_suffix(value: str) -> str:
    suffix = " language"
    if value.endswith(suffix):
        return value[: -len(suffix)].rstrip()
    return value


def _finalize_model_config(config: TranslateGemmaConfig) -> TranslateGemmaConfig:
    if not config.model_map:
        raise ValueError("model_map must not be empty.")
    return config


_ENV_OVERRIDE_MAP = {
    "TRANSLATEGEMMA_MODEL_IDLE_TIMEOUT": ("model_idle_timeout_seconds", int),
    "MAX_CONTEXT_TOKENS": ("max_context_tokens", int),
    "DEFAULT_MAX_OUTPUT_TOKENS": ("default_max_output_tokens", int),
    "MIN_INPUT_BUDGET": ("min_input_budget", int),
    "SRC_LANGUAGE_CODE": ("src_language_code", _normalize_lang_value),
    "DST_LANGUAGE_CODE": ("dst_language_code", _normalize_lang_value),
    "TRANSLATEGEMMA_VERBOSE": ("verbose_logging", lambda v: v.strip().lower() in _TRUE_VALUES),
}


def _load_yaml_data(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Configuration file {path} must contain a mapping")
    return data


def load_config() -> TranslateGemmaConfig:
    cfg_path = Path(os.environ.get("TRANSLATEGEMMA_CONFIG_FILE", DEFAULT_CONFIG_PATH))
    yaml_data = _load_yaml_data(cfg_path)
    config = TranslateGemmaConfig(**yaml_data)

    overrides: Dict[str, Any] = {}
    for env_key, (field, caster) in _ENV_OVERRIDE_MAP.items():
        raw = os.environ.get(env_key)
        if raw is None:
            continue
        overrides[field] = caster(raw)

    if overrides:
        config = config.copy(update=overrides)

    return _finalize_model_config(config)
