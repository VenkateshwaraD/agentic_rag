"""Chat model factory — Claude via langchain-anthropic, lazy singleton."""

from functools import lru_cache

from app.core.config import get_settings


@lru_cache
def get_llm():
    from langchain_anthropic import ChatAnthropic

    s = get_settings()
    return ChatAnthropic(
        model=s.anthropic_model,
        api_key=s.anthropic_api_key or None,
        max_tokens=1024,
        temperature=0,
    )
