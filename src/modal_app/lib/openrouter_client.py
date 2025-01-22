# %%
import os
from openai import OpenAI


def get_openrouter_client():
    """Get OpenRouter client with proper configuration"""
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        default_headers={
            "HTTP-Referer": "https://community-archive.org",
            "X-Title": "community-archive",
        },
    )


def query_llm(
    message: str,
    model: str = "meta-llama/llama-3.3-70b-instruct",
    max_tokens: int = 8000,
    temperature: float = 0.0,
) -> str:
    """Query LLM through OpenRouter

    Args:
        message: Prompt to send
        model: Model to use
        max_tokens: Max tokens in response
        temperature: Sampling temperature

    Returns:
        Model response text
    """
    client = get_openrouter_client()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": message}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content
