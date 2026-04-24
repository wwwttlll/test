from tts_cir.config import SamplingConfig
from tts_cir.metrics import bootstrap_mean_ci
from tts_cir.sampling import sample_hypotheses


def test_textual_sampling_uses_template_bank() -> None:
    hyps = sample_hypotheses(
        mod_text="make it elegant",
        ref_embedding=[0.1, 0.2, 0.3],
        k=4,
        cfg=SamplingConfig(mode="textual", seed=1),
    )
    assert len(hyps) == 4
    assert len({h.text for h in hyps}) == 4


def test_bootstrap_ci_is_ordered() -> None:
    mean, low, high = bootstrap_mean_ci([0.0, 1.0, 0.0, 1.0], n_bootstrap=100, seed=7)
    assert low <= mean <= high
