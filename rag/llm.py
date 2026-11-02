import os
from dataclasses import dataclass

from openai import OpenAI


@dataclass(frozen=True)
class LLMConfig:
    api_key: str | None
    base_url: str
    model: str
    mock: bool


def load_llm_config() -> LLMConfig:
    mock = os.getenv("MOCK_LLM", "").strip() in {"1", "true", "yes"}
    return LLMConfig(
        api_key=os.getenv("OPENAI_API_KEY") or None,
        base_url=os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1"),
        model=os.getenv("MODEL", "openai/gpt-4o-mini"),
        mock=mock,
    )


SYSTEM_PROMPT = """You are a helpful FAQ assistant for a business.
Answer ONLY using the provided context excerpts from the company's documents.
If the context does not contain enough information to answer, reply with exactly:
I don't know.
Be concise and friendly. When you use facts from the context, you may reference them naturally.
Do not invent policies, prices, or features not stated in the context."""


def generate_answer(question: str, context_block: str, config: LLMConfig | None = None) -> str:
    config = config or load_llm_config()

    if config.mock:
        if not context_block.strip():
            return "I don't know."
        return (
            f"[Mock mode] Based on our documents: {context_block[:500]}"
            + ("…" if len(context_block) > 500 else "")
        )

    if not config.api_key:
        return (
            "Please add OPENAI_API_KEY to your environment (see .env.example). "
            "Or set MOCK_LLM=1 to try the demo without an API key."
        )

    client = OpenAI(api_key=config.api_key, base_url=config.base_url)
    user_content = f"""Context from company documents:
---
{context_block}
---

Customer question: {question}

Answer using only the context above."""

    response = client.chat.completions.create(
        model=config.model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0.2,
        max_tokens=512,
    )
    return (response.choices[0].message.content or "").strip()
