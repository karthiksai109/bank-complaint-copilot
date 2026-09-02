"""Train the complaint classifier on sample texts and persist it with joblib.

In a real deployment this would train on historical CFPB complaint records
(the CFPB publishes a public complaint database). For the repo I keep a small
synthetic set so training runs in a second and needs no dataset download.
"""
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

MODEL_PATH = Path(__file__).parent / "classifier.joblib"

SAMPLE_DATA = {
    "fraud_unauthorized_transactions": [
        "There are charges on my debit card that I never made, someone stole my card info",
        "I found unauthorized ATM withdrawals from my account last weekend",
        "My account was drained by purchases I do not recognize, I think I was hacked",
        "Someone used my card number online and my bank has not refunded me",
        "Fraudulent checks cleared against my checking account",
        "I never opened this credit card, it must be identity theft",
        "Unauthorized Zelle transfers emptied my savings overnight",
        "A merchant charged me twice and calls it a duplicate fraud case",
        "My online banking password was changed by somebody else and money is gone",
        "I reported my card stolen but more fraudulent charges posted after that",
    ],
    "overdraft_fees": [
        "I was charged an overdraft fee even though my balance was positive at the time",
        "The bank reordered my transactions to trigger multiple overdraft fees",
        "I opted out of overdraft coverage but still got charged $35",
        "Three overdraft fees hit in one day for small debit purchases",
        "They charged a fee while my paycheck deposit was pending",
        "The overdraft protection transfer did not work and I was still penalized",
        "I believe these overdraft charges are unfair and want them refunded",
        "Continuous overdraft daily fees kept accruing without notice",
        "An overdraft fee appeared on a transaction that was approved against available balance",
        "Authorize positive settle negative is costing me hundreds in fees",
    ],
    "payment_app_p2p": [
        "I sent money on Zelle to the wrong person and the bank refuses to help",
        "A scammer tricked me into a Zelle payment for tickets that never arrived",
        "My Zelle payment is pending for days and the recipient never got the funds",
        "I was scammed into authorizing a peer to peer transfer",
        "Zelle shows the payment completed but the seller disappeared",
        "I cannot cancel a Zelle transfer made minutes ago",
        "Someone impersonated the bank and had me send a Zelle for verification",
        "Duplicate Zelle payment was deducted from my account",
    ],
    "account_access": [
        "My online banking login stopped working and I am locked out of my account",
        "The app keeps crashing when I try to deposit a check by mobile",
        "Two factor authentication codes never arrive to my phone",
        "I cannot reset my password, the reset link is broken",
        "My account was frozen without any explanation and I cannot pay bills",
        "The mobile app shows the wrong balance for two days now",
        "Bank locked my debit card without telling me while I was traveling",
        "I have been unable to reach anyone to unlock my online profile",
    ],
    "mortgage_servicing": [
        "My mortgage servicer said I was late even though I paid on time",
        "Escrow analysis raised my monthly payment with no explanation",
        "They misapplied my extra principal payment to interest instead",
        "Foreclosure notice arrived while my loan modification was under review",
        "My PMI should have been cancelled months ago at 80 percent LTV",
        "The servicer lost my payoff documents and charged me more interest",
        "Force placed flood insurance was added even though I have coverage",
        "Credit reporting shows my mortgage 30 days late which is wrong",
    ],
    "credit_card_billing": [
        "My credit card APR went up with no notice at all",
        "I was charged interest on a balance I paid in full",
        "A returned purchase never got credited back to my card",
        "The annual fee was billed after I had already cancelled the card",
        "Promotional zero percent APR was not honored on my balance transfer",
        "A merchant refund is showing as a charge instead of a credit",
        "My card statement includes a membership fee I never agreed to",
        "Billing errors keep repeating on my statement every month",
    ],
    "wire_transfers": [
        "My international wire has been stuck for two weeks with no update",
        "The bank charged me a wire fee that was not disclosed",
        "A wire I sent arrived short because of intermediary fees nobody told me about",
        "I asked to recall a wire and heard nothing back for days",
        "Incoming wire to my business account is being held and payroll is late",
        "The SWIFT details I provided were entered wrong by the branch staff",
        "Wire transfer confirmation shows the wrong beneficiary name",
        "Exchange rate applied to my wire was far worse than quoted",
    ],
    "loan_servicing": [
        "My auto loan payment was auto drafted twice this month",
        "They reported my personal loan as delinquent while I was on deferment",
        "Student loan forbearance was ended without notice and they demand payment",
        "Payoff quote for my car loan is higher than the balance shown online",
        "The lender never applied my insurance refund to the loan principal",
        "Auto loan title was never sent after I paid off the vehicle",
        "My line of credit was closed without warning and hurt my credit score",
        "Extra payments are not reducing my loan principal as promised",
    ],
}

# categories where a slow or wrong answer has regulatory / financial risk,
# bump these to high priority at triage time
HIGH_PRIORITY = {"fraud_unauthorized_transactions", "payment_app_p2p", "mortgage_servicing"}


def build_frame():
    rows = [(text, category) for category, texts in SAMPLE_DATA.items() for text in texts]
    texts = [t for t, _ in rows]
    labels = [c for _, c in rows]
    return texts, labels


def train():
    texts, labels = build_frame()
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=1000, C=10.0)),
    ])
    pipe.fit(texts, labels)
    joblib.dump(pipe, MODEL_PATH)
    print(f"trained on {len(texts)} samples across {len(set(labels))} categories -> {MODEL_PATH}")


if __name__ == "__main__":
    train()
