from src.setfit import __version__, SetFitClassifier


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
