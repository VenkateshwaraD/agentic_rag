"""Chat model factory — Claude via langchain-anthropic, lazy singleton.

Author: venkateshwara doijode
"""

from functools import lru_cache

from app.core.config import get_settings


@lru_cache
def get_llm():
    """Return the cached ``ChatAnthropic`` client configured from settings."""
    from langchain_anthropic import ChatAnthropic

    settings = get_settings()
    return ChatAnthropic(
        model=settings.anthropic_model,
        api_key=settings.anthropic_api_key or None,
        max_tokens=1024,
        temperature=0,
        timeout=float(settings.llm_timeout_s),
    )
