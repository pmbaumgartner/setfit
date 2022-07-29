from src.setfit import __version__, SetFitClassifier
from sklearn.exceptions import NotFittedError
import pytest
import numpy as np


def test_version():
    assert __version__ == "0.1.5"


def test_e2e():
    docs = ["yay", "boo", "yes", "no", "yeah"]
    labels = [1, 0, 1, 0, 1]

    # takes a sentence-transformers model
    clf = SetFitClassifier("paraphrase-MiniLM-L3-v2")
    # fine-tunes embeddings + trains logistic regression head
    clf.fit(docs, labels)

    preds = clf.predict(["affirmitive", "negative"])
    assert preds.shape == (2,)

    pproba = clf.predict_proba(["affirmitive", "negative"])
    assert pproba.shape == (2, 2)


def test_notfitted_error(tmp_path):
    docs = ["yay", "boo", "yes", "no", "yeah"]
    clf = SetFitClassifier("paraphrase-MiniLM-L3-v2")

    with pytest.raises(NotFittedError):
        clf.predict(docs)

    with pytest.raises(NotFittedError):
        clf.predict_proba(docs)

    with pytest.raises(NotFittedError):
        clf.save(tmp_path)


def test_save_load(tmp_path):
    docs = ["yay", "boo", "yes", "no", "yeah"]
    labels = [1, 0, 1, 0, 1]
    clf = SetFitClassifier("paraphrase-MiniLM-L3-v2")
    clf.fit(docs, labels)
    p1 = clf.predict(docs)
    clf.save(tmp_path)
    clf2 = SetFitClassifier.load(tmp_path)
    p2 = clf2.predict(docs)
    assert np.array_equal(p1, p2)


def test_get_params():
    # required for GridSearchCV
    clf = SetFitClassifier("paraphrase-MiniLM-L3-v2")
    assert clf.get_params()


def test_single_example_no_loop():
    docs = ["yes", "no"]
    labels = [1, 0]

    # takes a sentence-transformers model
    clf = SetFitClassifier("paraphrase-MiniLM-L3-v2")
    # fine-tunes embeddings + trains logistic regression head
    clf.fit(docs, labels)

    preds = clf.predict(["affirmitive", "negative"])
    assert preds.shape == (2,)

    pproba = clf.predict_proba(["affirmitive", "negative"])
    assert pproba.shape == (2, 2)


def test_multilabel():
    docs = ["yay", "boo", "yes", "no", "yeah", "maybe", "don't know"]
    labels = [1, 0, 1, 0, 1, 2, 2]

    n_labels = len(set(labels))

    # takes a sentence-transformers model
    clf = SetFitClassifier("paraphrase-MiniLM-L3-v2")
    # fine-tunes embeddings + trains logistic regression head
    clf.fit(docs, labels)

    pred_examples = ["affirmitive", "negative", "possibly", "uncertain"]
    n_preds = len(pred_examples)
    preds = clf.predict(["affirmitive", "negative", "possibly", "uncertain"])
    assert preds.shape == (n_preds,)

    pproba = clf.predict_proba(["affirmitive", "negative", "possibly", "uncertain"])
    assert pproba.shape == (n_preds, n_labels)
