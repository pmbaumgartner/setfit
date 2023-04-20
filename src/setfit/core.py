from pathlib import Path
import random
from collections import defaultdict
from itertools import chain, groupby
from typing import Any, List, Optional, Union

import joblib
import numpy as np
import torch
from sentence_transformers import InputExample, SentenceTransformer, losses
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader

StrOrPath = Union[Path, str]


def check_fitted(model):
    if not getattr(model, "fitted", False):
        raise NotFittedError(
            "This SetFitClassifier instance is not fitted yet."
            " Call 'fit' with appropriate arguments before saving this estimator."
        )


def generate_sentence_pair_batch(
    sentences: List[str], labels: List[float], alpha: float = 0.0
) -> List[InputExample]:
    """Generate a batch of sentence pairs of labeled data.
    Args:
        sentences (List[str]): Input Sentences
        labels (List[float]): Input Labels
        alpha (float, optional): Soft labeling regularization to apply
          (i.e. same label = 1 - alpha). Defaults to 0.0.
    Returns:
        List[InputExample]: Pairs of sentences, len=(2*sentences),
          with a label of 0 if different-label, 1 if same label.
    """
    if alpha > 0.5 or alpha < 0.0:
        raise ValueError(
            f"`alpha` must be between 0 and 0.5. "
            f"You passed {alpha}. "
            "`alpha` > 0.5 will invert the label."
        )
    # 7x faster than original implementation on small data,
    # 14x faster on 10000 examples
    pairs = []
    sent_lookup = defaultdict(list)
    for label, grouper in groupby(
        sorted(
            ((sent, label) for sent, label in zip(sentences, labels)),
            key=lambda x: x[1],
        ),
        key=lambda x: x[1],
    ):
        sent_lookup[label].extend(list(i[0] for i in grouper))
    neg_lookup = {}
    for current_label in sent_lookup:
        negative_options = list(
            chain.from_iterable(
                [
                    sentences
                    for label, sentences in sent_lookup.items()
                    if label != current_label
                ]
            )
        )
        neg_lookup[current_label] = negative_options

    for current_sentence, current_label in zip(sentences, labels):
        # Choose itself? Seems wrong.
        positive_pair = random.choice(sent_lookup[current_label])
        while positive_pair == current_sentence:
            positive_pair = random.choice(sent_lookup[current_label])

        negative_pair = random.choice(neg_lookup[current_label])
        pairs.append(
            InputExample(texts=[current_sentence, positive_pair], label=1.0 - alpha)
        )
        pairs.append(
            InputExample(texts=[current_sentence, negative_pair], label=0.0 + alpha)
        )
    return pairs


def generate_multiple_sentence_pairs(
    sentences: List[str], labels: List[float], iter: int = 1, alpha: float = 0.0
) -> List[InputExample]:
    """Generate pairs of match/non-match sentences
    Args:
        sentences (List[str]): Input Sentences
        labels (List[float]): Input Labels
        iter (int, optional): Number of passes over the data to generate pairs.
          Defaults to 1.
        alpha (float, optional): Soft labeling regularization to apply
          (i.e. same label = 1 - alpha). Defaults to 0.0.
    Returns:
        List[InputExample]: Pairs of sentences, len=(2*sentences),
          with a label of 0 if different-label, 1 if same label.
    """

    all_pairs = []
    for _ in range(iter):
        all_pairs.extend(generate_sentence_pair_batch(sentences, labels, alpha=alpha))
    return all_pairs


class SetFitClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        model: str,
        classifier_head: Optional[Any] = None,
        loss=losses.CosineSimilarityLoss,
        random_state: int = 1234,
    ):
        random.seed(random_state)
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        self.random_state = random_state
        self.model = SentenceTransformer(model)
        if classifier_head is None:
            self.classifier_head = LogisticRegression()
        else:
            self.classifier_head = classifier_head()
        self.loss = loss(self.model)
        self.fitted = False

    def fit(
        self,
        X,
        y,
        data_iter: int = 5,
        train_iter: int = 1,
        batch_size: int = 16,
        warmup_steps: int = 10,
        show_progress_bar: bool = True,
    ):
        train_examples = generate_multiple_sentence_pairs(X, y, data_iter)
        train_dataloader = DataLoader(
            train_examples,
            shuffle=True,
            batch_size=batch_size,
            generator=torch.Generator(device=self.model.device),
        )
        self.model.fit(
            train_objectives=[(train_dataloader, self.loss)],
            epochs=train_iter,
            warmup_steps=warmup_steps,
            show_progress_bar=show_progress_bar,
        )

        X_train = self.model.encode(X)
        self.classifier_head.fit(X_train, y)
        self.fitted = True

    def predict(self, X, y=None):
        check_fitted(self)
        X_embed = self.model.encode(X)
        preds = self.classifier_head.predict(X_embed)
        return preds

    def predict_proba(self, X, y=None):
        check_fitted(self)
        X_embed = self.model.encode(X)
        preds = self.classifier_head.predict_proba(X_embed)
        return preds

    def save(
        self,
        path: StrOrPath,
        model_name: Optional[str] = None,
        create_model_card: bool = False,
    ):
        check_fitted(self)
        self.model.save(str(path), model_name, create_model_card)
        joblib.dump(self.classifier_head, Path(path) / "classifier.pkl")

    @classmethod
    def load(cls, path: StrOrPath):
        setfit = SetFitClassifier(str(path))
        setfit.classifier_head = joblib.load(Path(path) / "classifier.pkl")
        setfit.fitted = True
        return setfit
