from pathlib import Path
import random
from collections import defaultdict
from itertools import chain, groupby
from typing import List, Optional, Union, cast

import joblib
import numpy as np
import torch
from sentence_transformers import InputExample, SentenceTransformer, losses
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from torch import nn

StrOrPath = Union[Path, str]


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
            "`alpha` > 0.5 will inverts the label."
        )
    # 7x faster than original implementation on small data,
    # 14x faster on 10000 examples
    pairs = []
    sent_lookup = defaultdict(list)
    for label, grouper in groupby(
        ((sent, label) for sent, label in zip(sentences, labels)), key=lambda x: x[1]
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
        model: SentenceTransformer = SentenceTransformer("all-MiniLM-L6-v2"),
        classifier_head: ClassifierMixin = LogisticRegression(),
        loss: nn.Module = losses.CosineSimilarityLoss,
        random_state: int = 1234,
    ):
        self.random_state = random_state
        random.seed(random_state)
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        self.model = model
        self.classifier_head = classifier_head
        # TODO: Loss should not create object
        self.loss = loss(self.model)
        self.fitted = False

    def fit(
        self,
        X: List[str],
        y: List[float],
        alpha: float = 0.0,
        data_iter: int = 5,
        train_iter: int = 1,
        batch_size: int = 16,
        warmup_steps: int = 10,
        show_progress_bar: bool = True,
    ):
        """Fit the model

        Args:
            X (List[str]): Documents to fit on
            y (List[float]): Labels for documents (can be multiclass)
            alpha (float, optional): Soft labeling regularization to apply
              (i.e. same label = 1 - alpha). Defaults to 0.0.
            data_iter (int, optional): How many iterations to create sentence pairs.
              One iteration means a match and non-match per example. Defaults to 5.
            train_iter (int, optional): Train epochs. Defaults to 1.
            batch_size (int, optional): Batch size for training. Defaults to 16.
            warmup_steps (int, optional): Warmup steps for training. Defaults to 10.
            show_progress_bar (bool, optional): show progress bar while training.
              Defaults to True.
        """
        train_examples = generate_multiple_sentence_pairs(
            X, y, iter=data_iter, alpha=alpha
        )
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
        X_embed = self.model.encode(X, convert_to_numpy=True)
        X_embed = cast(np.ndarray, X_embed)
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
        model = SentenceTransformer(str(path))
        setfit = SetFitClassifier(model)
        setfit.classifier_head = joblib.load(Path(path) / "classifier.pkl")
        setfit.fitted = True
        return setfit
