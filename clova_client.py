"""
CLOVA Studio client helpers (Chat Completions v3).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence

import requests

from ss_utils import clova_studio_headers


DEFAULT_CLOVA_STUDIO_BASE_URL = "https://clovastudio.stream.ntruss.com"


@dataclass(frozen=True)
class ClovaChatMessage:
    role: str
    content: str


def load_clova_studio_base_url() -> str:
    return (os.getenv("CLOVA_STUDIO_BASE_URL") or DEFAULT_CLOVA_STUDIO_BASE_URL).rstrip("/")


def load_clova_studio_model() -> str | None:
    value = (os.getenv("CLOVA_STUDIO_MODEL") or "").strip()
    return value or None


def _messages_to_payload(messages: Sequence[ClovaChatMessage | Mapping[str, str]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for m in messages:
        if isinstance(m, ClovaChatMessage):
            out.append({"role": m.role, "content": m.content})
        else:
            out.append({"role": str(m["role"]), "content": str(m["content"])})
    return out


def clova_chat_completions_v3(
    *,
    messages: Sequence[ClovaChatMessage | Mapping[str, str]],
    model: str | None = None,
    temperature: float | None = 0.7,
    top_p: float | None = None,
    thinking: str | None = "medium",
    timeout_s: float = 120.0,
    extra: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Call CLOVA Studio Chat Completions v3 API.

    Endpoint:
      POST {CLOVA_STUDIO_BASE_URL}/v3/chat-completions/{model}

    Auth headers (per docs):
      Authorization: Bearer <CLOVA_STUDIO_API_KEY>
      Content-Type: application/json

    Args:
        messages: Conversation messages
        model: Model name (defaults to CLOVA_STUDIO_MODEL env var)
        temperature: Sampling temperature (0.0-1.0)
        top_p: Top-p sampling parameter
        thinking: Thinking effort level for HCX-007 reasoning model.
                  Options: "short", "medium", "deep", or None to disable.
                  Default: "medium" for balanced reasoning.
        timeout_s: Request timeout in seconds
        extra: Additional payload fields

    Notes:
    - HCX-007 is a reasoning model with built-in thinking capability.
    - Thinking enables the model to reason before responding (like o1/o3).
    """
    resolved_model = (model or load_clova_studio_model() or "").strip()
    if not resolved_model:
        raise RuntimeError(
            "CLOVA Studio model missing. Set CLOVA_STUDIO_MODEL in .env or pass model=... explicitly."
        )

    payload: Dict[str, Any] = {
        "messages": _messages_to_payload(messages),
    }
    if temperature is not None:
        payload["temperature"] = float(temperature)
    if top_p is not None:
        payload["topP"] = float(top_p)
    if thinking is not None:
        payload["thinking"] = {"type": thinking}
    if extra:
        payload.update(dict(extra))

    url = f"{load_clova_studio_base_url()}/v3/chat-completions/{resolved_model}"
    resp = requests.post(url, headers=clova_studio_headers(), json=payload, timeout=timeout_s)
    resp.raise_for_status()
    return resp.json()

