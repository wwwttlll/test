from tts_cir.hf_pipeline import HFCIRRetriever


def test_hf_retriever_symbol_available():
    assert HFCIRRetriever is not None
