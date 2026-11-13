"""Langfuse tracing — graceful no-op when keys aren't configured.

Langfuse v3+ SDK reads LANGFUSE_SECRET_KEY / LANGFUSE_HOST from env; we set
them from Settings before constructing the handler. Falls back to the v2
`langfuse.callback` path on older SDKs. The request's trace_id propagates
to audit rows and the X-Trace-Id response header.
"""

import os
from typing import Any

from app.core.config import get_settings
from app.core.logging import get_logger

log = get_logger(__name__)


def build_callback_handler() -> tuple[Any | None, str | None]:
    """Return (handler, trace_name) or (None, None) if tracing disabled."""
    settings = get_settings()
    if not settings.langfuse_enabled:
        return None, None
    try:
        os.environ.setdefault("LANGFUSE_SECRET_KEY", settings.langfuse_secret_key)
        os.environ.setdefault("LANGFUSE_HOST", settings.langfuse_host)
        try:
            from langfuse.langchain import CallbackHandler  # v3+

            handler = CallbackHandler(public_key=settings.langfuse_public_key)
        except ImportError:
            from langfuse.callback import CallbackHandler as V2Handler

            handler = V2Handler(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_host,
            )
        return handler, "copilot-answer"
    except Exception as exc:
        log.warning("tracing.disabled", reason=str(exc))
        return None, None


def graph_config(handler: Any | None, request_id: str, user_id: str) -> dict:
    """LangChain runnable config wiring the Langfuse callback + metadata."""
    config: dict = {
        "run_name": "copilot-answer",
        "metadata": {"request_id": request_id, "user_id": user_id},
    }
    if handler is not None:
        config["callbacks"] = [handler]
    return config
