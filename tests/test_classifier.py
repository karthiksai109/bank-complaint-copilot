from services.api.model.classifier import ComplaintClassifier


def test_fraud_is_high_priority():
    clf = ComplaintClassifier()
    out = clf.predict("Someone withdrew money from my account at an ATM that I never visited")
    assert out["category"] == "fraud_unauthorized_transactions"
    assert out["priority"] == "high"
    assert 0 < out["confidence"] <= 1


def test_overdraft_text_classifies_correctly():
    clf = ComplaintClassifier()
    out = clf.predict("The bank charged me three overdraft fees in one day for tiny purchases")
    assert out["category"] == "overdraft_fees"


def test_wire_text_classifies_correctly():
    clf = ComplaintClassifier()
    out = clf.predict("My international wire has been missing for two weeks and nobody can trace it")
    assert out["category"] == "wire_transfers"
