from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class SamplingConfig:
    mode: str = "textual"  # textual | latent
    temperature: float = 0.7
    noise_scale: float = 0.1
    seed: int = 42


@dataclass(frozen=True)
class ScoringConfig:
    w_obj: float = 1.0
    w_attr: float = 1.0
    w_style: float = 0.5
    lambda_neg: float = 0.5


@dataclass(frozen=True)
class ExperimentConfig:
    top_n: int = 400
    k_values: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32])
    sample_repeats: int = 1
    seeds: List[int] = field(default_factory=lambda: [42, 43, 44])
    bootstrap_samples: int = 1000
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
