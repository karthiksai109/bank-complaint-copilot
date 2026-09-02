from services.rag.retriever import PolicyRetriever


def test_retriever_finds_reg_e_section():
    r = PolicyRetriever()
    hits = r.search("how long do we have to investigate an unauthorized ATM withdrawal")
    top = hits[0][0]
    assert top.source == "reg_e_error_resolution.md"


def test_retriever_finds_zelle_section():
    r = PolicyRetriever()
    hits = r.search("customer sent money to a scammer with Zelle can we reverse it")
    top = hits[0][0]
    assert top.source == "zelle_p2p_payments.md"


def test_every_doc_loads():
    r = PolicyRetriever()
    sources = {c.source for c in r.chunks}
    assert len(sources) == 5
