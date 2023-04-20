from setfit import SetFitClassifier
from time import time
from tabulate import tabulate

# Via Stack Overflow
# https://stackoverflow.com/questions/11130156/suppress-stdout-stderr-print-from-python-functions
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from os import devnull


@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(devnull, "w") as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)


models = [
    "all-mpnet-base-v2",
    # "gtr-t5-xxl",
    # "gtr-t5-xl",
    # "sentence-t5-xxl",
    "gtr-t5-large",
    "all-mpnet-base-v1",
    "multi-qa-mpnet-base-dot-v1",
    "multi-qa-mpnet-base-cos-v1",
    "all-roberta-large-v1",
    # "sentence-t5-xl",
    "all-distilroberta-v1",
    "all-MiniLM-L12-v1",
    "all-MiniLM-L12-v2",
    "multi-qa-distilbert-dot-v1",
    "multi-qa-distilbert-cos-v1",
    "gtr-t5-base",
    "sentence-t5-large",
    "all-MiniLM-L6-v2",
    "multi-qa-MiniLM-L6-cos-v1",
    "all-MiniLM-L6-v1",
    "paraphrase-mpnet-base-v2",
    "msmarco-bert-base-dot-v5",
    "multi-qa-MiniLM-L6-dot-v1",
    "sentence-t5-base",
    "msmarco-distilbert-base-tas-b",
    "msmarco-distilbert-dot-v5",
    "paraphrase-distilroberta-base-v2",
    "paraphrase-MiniLM-L12-v2",
    "paraphrase-multilingual-mpnet-base-v2",
    "paraphrase-TinyBERT-L6-v2",
    "paraphrase-MiniLM-L6-v2",
    "paraphrase-albert-small-v2",
    "paraphrase-multilingual-MiniLM-L12-v2",
    "paraphrase-MiniLM-L3-v2",
    "distiluse-base-multilingual-cased-v1",
    "distiluse-base-multilingual-cased-v2",
]

docs = [
    # positives
    "yes",
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
    # negatives
    "no",
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

labels = [1.0] * 11 + [0.0] * 11

eval_data = []
for model in models[-4:-3]:
    with suppress_stdout_stderr():
        # Don't output the model download info
        clf = SetFitClassifier(model)
    start = time()
    clf.fit(docs, labels, show_progress_bar=False)
    train_duration = time() - start
    preds = clf.predict(["affirmitive", "negative"])
    pproba = clf.predict_proba(["affirmitive", "negative"])
    data = {
        "model": model,
        "train duration": f"{train_duration:.2f}",
        "affirmitive correct": bool(preds[0] == 1),
        "negative correct": bool(preds[1] == 0),
        "affirmitive prob positive": f"{float(pproba[0, 1]):.3f}",
        "negative prob negative": f"{float(pproba[1, 0]):.3f}",
    }
    eval_data.append(data)

print(tabulate(eval_data, headers="keys", tablefmt="github"))
