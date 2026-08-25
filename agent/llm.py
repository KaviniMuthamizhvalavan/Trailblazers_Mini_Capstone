import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

def get_llm():
    """Get the configured LLM instance using environment settings."""
    # Prioritize NEW_OPENAI_API_KEY if configured
    new_key = os.getenv("NEW_OPENAI_API_KEY", "").strip()
    if new_key:
        api_key = new_key
        base_url = "https://api.openai.com/v1"
        # Temporarily remove OPENAI_BASE_URL to avoid OpenAI SDK fallback routing to gateway
        orig_base_url = os.environ.get("OPENAI_BASE_URL")
        if "OPENAI_BASE_URL" in os.environ:
            del os.environ["OPENAI_BASE_URL"]
    else:
        api_key = os.getenv("OPENAI_API_KEY", "")
        base_url = os.getenv("OPENAI_BASE_URL", "")
        orig_base_url = None

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    kwargs = {
        "model": model,
        "api_key": api_key,
        "temperature": 0,
        "request_timeout": 60.0,
        "max_retries": 10,
        "max_tokens": 4096,
    }
    if base_url:
        kwargs["base_url"] = base_url

    try:
        llm = ChatOpenAI(**kwargs)
    finally:
        # Restore environment variable so other components aren't affected
        if orig_base_url is not None:
            os.environ["OPENAI_BASE_URL"] = orig_base_url

    return llm
