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


def generate_sentence_pair_batch(
    sentences: List[str], labels: List[float]
) -> List[InputExample]:
    # 7x faster than original implementation on small data,
    # 14x faster on 10000 examples
    pairs = []
    sent_lookup = defaultdict(list)
    single_example = {}
    for label, grouper in groupby(
        ((s, l) for s, l in zip(sentences, labels)), key=lambda x: x[1]
    ):
        sent_lookup[label].extend(list(i[0] for i in grouper))
        single_example[label] = len(sent_lookup[label]) == 1
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
        positive_pair = random.choice(sent_lookup[current_label])
        if not single_example[current_label]:
            # choosing itself as a matched pair seems wrong,
            # but we need to account for the case of 1 positive example
            # so as long as there's not a single positive example,
            # we'll reselect the other item in the pair until it's different
            while positive_pair == current_sentence:
                positive_pair = random.choice(sent_lookup[current_label])

        negative_pair = random.choice(neg_lookup[current_label])
        pairs.append(InputExample(texts=[current_sentence, positive_pair], label=1.0))
        pairs.append(InputExample(texts=[current_sentence, negative_pair], label=0.0))

    return pairs


def generate_multiple_sentence_pairs(
    sentences: List[str], labels: List[float], iter: int = 1
):
    all_pairs = []
    for _ in range(iter):
        all_pairs.extend(generate_sentence_pair_batch(sentences, labels))
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
        if not self.fitted:
            raise NotFittedError(
                "This SetFitClassifier instance is not fitted yet."
                " Call 'fit' with appropriate arguments before using this estimator."
            )
        X_embed = self.model.encode(X)
        preds = self.classifier_head.predict(X_embed)
        return preds

    def predict_proba(self, X, y=None):
        if not self.fitted:
            raise NotFittedError(
                "This SetFitClassifier instance is not fitted yet."
                " Call 'fit' with appropriate arguments before using this estimator."
            )
        X_embed = self.model.encode(X)
        preds = self.classifier_head.predict_proba(X_embed)
        return preds

    def save(
        self,
        path: StrOrPath,
        model_name: Optional[str] = None,
        create_model_card: bool = False,
    ):
        if not self.fitted:
            raise NotFittedError(
                "This SetFitClassifier instance is not fitted yet."
                " Call 'fit' with appropriate arguments before saving this estimator."
            )
        self.model.save(str(path), model_name, create_model_card)
        joblib.dump(self.classifier_head, Path(path) / "classifier.pkl")

    @classmethod
    def load(cls, path: StrOrPath):
        setfit = SetFitClassifier(str(path))
        setfit.classifier_head = joblib.load(Path(path) / "classifier.pkl")
        setfit.fitted = True
        return setfit
