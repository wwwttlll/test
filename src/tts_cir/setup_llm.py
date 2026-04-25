"""Setup LLM configuration for SPVI."""
import os
import sys


def setup_llm(api_key: str = None, base_url: str = None, model: str = None) -> None:
    """Configure LLM API settings.
    
    Args:
        api_key: API key (or set LLM_API_KEY env var)
        base_url: API base URL (or set LLM_BASE_URL env var)
        model: Model name (or set LLM_MODEL env var)
    """
    if api_key:
        os.environ["LLM_API_KEY"] = api_key
        print(f"✓ LLM_API_KEY set")
    else:
        api_key = os.environ.get("LLM_API_KEY", "")
        if not api_key:
            print("⚠ LLM_API_KEY not set")
    
    if base_url:
        os.environ["LLM_BASE_URL"] = base_url
        print(f"✓ LLM_BASE_URL set to: {base_url}")
    else:
        base_url = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
        print(f"  LLM_BASE_URL: {base_url}")
    
    if model:
        os.environ["LLM_MODEL"] = model
        print(f"✓ LLM_MODEL set to: {model}")
    else:
        model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
        print(f"  LLM_MODEL: {model}")


def test_connection() -> bool:
    """Test if LLM API is working."""
    try:
        from .llm_client import generate_structured_proposals
        result = generate_structured_proposals(
            ref_caption="a red shirt",
            mod_text="make it blue",
            k=2,
        )
        print(f"✓ LLM connection successful: {len(result)} proposals generated")
        return True
    except Exception as e:
        print(f"✗ LLM connection failed: {e}")
        return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m tts_cir.setup_llm <api_key> [base_url] [model]")
        print("  Or set environment variables:")
        print("    export LLM_API_KEY=your_key")
        print("    export LLM_BASE_URL=https://api.openai.com/v1")
        print("    export LLM_MODEL=gpt-4o-mini")
        sys.exit(1)
    
    api_key = sys.argv[1]
    base_url = sys.argv[2] if len(sys.argv) > 2 else None
    model = sys.argv[3] if len(sys.argv) > 3 else None
    
    setup_llm(api_key, base_url, model)
    test_connection()
