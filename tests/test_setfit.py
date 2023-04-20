import numpy as np
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.utils.estimator_checks import check_estimator
from sentence_transformers import SentenceTransformer
from src.setfit import SetFitClassifier, __version__


def test_version():
    assert __version__ == "0.1.3"


def test_e2e():
    docs = [
        "aye",
        "certainly",
        "definitely",
        "indeed",
        "ok",
        "roger",
        "right",
        "sure",
        "yep",
        "yeah",
        "never",
        "nay",
        "nope",
        "not",
        "denial",
        "refusal",
        "veto",
        "decline",
        "nix",
        "nah",
    ]

    labels = [1.0] * 10 + [0.0] * 10

    # takes a sentence-transformers model
    clf = SetFitClassifier()
    # fine-tunes embeddings + trains logistic regression head
    clf.fit(docs, labels)

    preds = clf.predict(["affirmitive", "negative"])
    assert preds.shape == (2,)
    print(preds)

    pproba = clf.predict_proba(["affirmitive", "negative"])
    assert pproba.shape == (2, 2)
    print(pproba)


def test_notfitted_error(tmp_path):
    docs = ["yay", "boo", "yes", "no", "yeah"]
    clf = SetFitClassifier()

    with pytest.raises(NotFittedError):
        clf.predict(docs)

    with pytest.raises(NotFittedError):
        clf.predict_proba(docs)

    with pytest.raises(NotFittedError):
        clf.save(tmp_path)


def test_save_load(tmp_path):
    docs = ["yay", "boo", "yes", "no", "yeah"]
    labels = [1, 0, 1, 0, 1]
    clf = SetFitClassifier()
    clf.fit(docs, labels)
    p1 = clf.predict(docs)
    clf.save(tmp_path)
    clf2 = SetFitClassifier.load(tmp_path)
    p2 = clf2.predict(docs)
    assert np.array_equal(p1, p2)


# def test_sklearn_estimator():
#     check_estimator(SetFitClassifier())
