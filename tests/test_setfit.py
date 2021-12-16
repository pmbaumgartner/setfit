from src.setfit import __version__, SetFitClassifier
from sklearn.exceptions import NotFittedError
import pytest


def test_version():
    assert __version__ == "0.1.0"


def test_e2e():
    docs = ["yay", "boo", "yes", "no", "yeah"]
    labels = [1, 0, 1, 0, 1]

    # takes a sentence-transformers model
    clf = SetFitClassifier("paraphrase-MiniLM-L3-v2")
    # fine-tunes embeddings + trains logistic regression head
    clf.fit(docs, labels)

    preds = clf.predict(["affirmitive", "negative"])
    assert preds.shape == (2,)


def test_notfitted_error():
    docs = ["yay", "boo", "yes", "no", "yeah"]
    clf = SetFitClassifier("paraphrase-MiniLM-L3-v2")

    with pytest.raises(NotFittedError):
        clf.predict(docs)

    with pytest.raises(NotFittedError):
        clf.predict_proba(docs)
