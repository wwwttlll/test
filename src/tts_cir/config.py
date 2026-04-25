from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class SamplingConfig:
    mode: str = "textual"  # textual | latent | structured | paraphrase
    temperature: float = 0.7
    noise_scale: float = 0.1
    seed: int = 42
    k_proposals: int = 4  # Number of LLM proposals for structured mode


@dataclass(frozen=True)
class ScoringConfig:
    w_obj: float = 1.0
    w_attr: float = 1.0
    w_style: float = 0.5
    lambda_neg: float = 0.5


@dataclass(frozen=True)
class VerifierConfig:
    beta: float = 0.5  # Weight for preference score
    top_m: int = 20    # Only verify top-M candidates
    use_cache: bool = True


@dataclass(frozen=True)
class LLMConfig:
    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"
    timeout: int = 30


@dataclass(frozen=True)
class ExperimentConfig:
    top_n: int = 400
    k_values: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32])
    sample_repeats: int = 1
    seeds: List[int] = field(default_factory=lambda: [42, 43, 44])
    bootstrap_samples: int = 1000
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    verifier: VerifierConfig = field(default_factory=VerifierConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
