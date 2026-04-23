from tts_cir.config import ScoringConfig
from tts_cir.metrics import oracle_recall_at_k, recall_at_k
from tts_cir.retrieval import Candidate
from tts_cir.sampling import EditHypothesis
from tts_cir.scoring import structured_score


def test_recall_metrics():
    ranked = ["a", "b", "c"]
    assert recall_at_k(ranked, "a", 1) == 1.0
    assert recall_at_k(ranked, "c", 2) == 0.0
    assert oracle_recall_at_k([ranked, ["x", "y", "z"]], "c", 3) == 1.0


def test_structured_score_runs():
    ref = [1.0, 0.0, 0.0]
    cand = Candidate(image_id="img_1", embedding=[0.8, 0.1, 0.0])
    hyp = EditHypothesis(
        text="make it elegant",
        weights={"obj": 1.0, "attr": 0.8, "style": 1.0, "neg": 0.5},
        direction=[0.0, 1.0, 0.0],
    )
    score = structured_score(ref, cand, hyp, ScoringConfig())
    assert isinstance(score.total, float)
