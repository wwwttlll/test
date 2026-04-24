from tts_cir.baselines import BaselineQuery, run_wiser_like_experiment
from tts_cir.config import ExperimentConfig, SamplingConfig, ScoringConfig
from tts_cir.retrieval import Candidate


def test_wiser_like_runner_outputs_expected_rounds() -> None:
    queries = [
        BaselineQuery(
            query_id="q1",
            reference_embedding=[1.0, 0.0],
            modification_text="make it elegant",
            target_id="img_a",
        )
    ]
    candidates = [
        Candidate(image_id="img_a", embedding=[1.0, 0.0]),
        Candidate(image_id="img_b", embedding=[0.0, 1.0]),
    ]

    cfg = ExperimentConfig(
        top_n=2,
        sampling=SamplingConfig(mode="textual", noise_scale=0.01),
        scoring=ScoringConfig(),
    )

    results = run_wiser_like_experiment(queries, candidates, cfg, rounds_list=[1, 2])

    assert [r.rounds for r in results] == [1, 2]
    assert all(r.forward_passes > 0 for r in results)
    assert all(0.0 <= r.r1 <= 1.0 for r in results)
