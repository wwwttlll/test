"""LLM client for structured proposal generation."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from functools import lru_cache


LLM_PROPOSAL_PROMPT = """Given a reference image concept: "{ref_caption}"
Modify it so that the target: "{mod_text}"

Generate {K} diverse interpretations as JSON array:
[
  {{
    "canonical_instruction": "specific reformulated instruction",
    "must_keep_entities": ["entity1", "entity2"],
    "must_add_or_change_attributes": ["attr1"],
    "must_remove_or_avoid": ["avoid1"],
    "spatial_or_relation": "constraint or null"
  }}
]

Respond ONLY with the JSON array."""


LLM_VERIFIER_PROMPT = """Reference concept: "{ref_caption}"
Modification: "{mod_text}"

Which candidate better satisfies the modification?
Candidate A: {cand_a_desc}
Candidate B: {cand_b_desc}

Output JSON: {{"winner": "A|B|TIE", "reason": "brief justification", "confidence": 0.0-1.0}}"""


@dataclass
class LLMConfig:
    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"
    timeout: int = 30


def get_llm_config() -> LLMConfig:
    """Get LLM config from environment or defaults."""
    return LLMConfig(
        api_key=os.environ.get("LLM_API_KEY", ""),
        base_url=os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1"),
        model=os.environ.get("LLM_MODEL", "gpt-4o-mini"),
        timeout=int(os.environ.get("LLM_TIMEOUT", "30")),
    )


def set_llm_config(api_key: str, base_url: str = "https://api.openai.com/v1", model: str = "gpt-4o-mini") -> None:
    """Set LLM config programmatically."""
    os.environ["LLM_API_KEY"] = api_key
    os.environ["LLM_BASE_URL"] = base_url
    os.environ["LLM_MODEL"] = model


@lru_cache(maxsize=128)
def _call_llm(prompt: str, config_str: str) -> str:
    """Cached LLM call."""
    import urllib.request
    import urllib.error
    
    config = get_llm_config()
    if not config.api_key:
        raise ValueError("LLM_API_KEY not set. Call set_llm_config() or set environment variable.")
    
    data = {
        "model": config.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
    }
    
    req = urllib.request.Request(
        f"{config.base_url}/chat/completions",
        data=json.dumps(data).encode(),
        headers={
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    
    try:
        with urllib.request.urlopen(req, timeout=config.timeout) as resp:
            result = json.loads(resp.read())
            return result["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"LLM API error: {e.code} {e.read().decode()}") from e


def generate_structured_proposals(
    ref_caption: str,
    mod_text: str,
    k: int = 4,
) -> List[Dict[str, Any]]:
    """Generate K structured proposal interpretations."""
    prompt = LLM_PROPOSAL_PROMPT.format(ref_caption=ref_caption, mod_text=mod_text, K=k)
    response = _call_llm(prompt, f"{ref_caption}|{mod_text}|{k}")
    
    # Try to parse JSON
    try:
        # Find JSON array in response
        start = response.find("[")
        end = response.rfind("]") + 1
        if start >= 0 and end > start:
            proposals = json.loads(response[start:end])
            return proposals[:k]
    except json.JSONDecodeError:
        pass
    
    # Fallback: return simple paraphrase
    return [{"canonical_instruction": mod_text}] * k


def pairwise_verdict(
    ref_caption: str,
    mod_text: str,
    cand_a_id: str,
    cand_b_id: str,
    cand_a_desc: str = "",
    cand_b_desc: str = "",
) -> Dict[str, Any]:
    """Get pairwise preference judgment."""
    prompt = LLM_VERIFIER_PROMPT.format(
        ref_caption=ref_caption,
        mod_text=mod_text,
        cand_a_desc=cand_a_desc or f"candidate {cand_a_id}",
        cand_b_desc=cand_b_desc or f"candidate {cand_b_id}",
    )
    response = _call_llm(prompt, f"verifier|{ref_caption}|{mod_text}|{cand_a_id}|{cand_b_id}")
    
    try:
        start = response.find("{")
        end = response.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(response[start:end])
    except json.JSONDecodeError:
        pass
    
    return {"winner": "TIE", "reason": "parsing failed", "confidence": 0.5}


def paraphrase_simple(mod_text: str, k: int = 4) -> List[str]:
    """Simple paraphrase baseline (no structured output)."""
    prompt = f"Paraphrase this instruction in {k} different ways: \"{mod_text}\". Output JSON array of strings."
    response = _call_llm(prompt, f"paraphrase|{mod_text}|{k}")
    
    try:
        start = response.find("[")
        end = response.rfind("]") + 1
        if start >= 0 and end > start:
            return json.loads(response[start:end])[:k]
    except json.JSONDecodeError:
        pass
    
    return [mod_text] * k
