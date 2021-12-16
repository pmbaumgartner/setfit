import random

import numpy as np
import torch
from sentence_transformers import (
    InputExample,
    SentenceTransformer,
    losses,
    models,
)
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import NotFittedError
from torch.utils.data import DataLoader


def sentence_pairs_generation(sentences, labels, pairs):
    # initialize two empty lists to hold the (sentence, sentence) pairs and
    # labels to indicate if a pair is positive or negative

    numClassesList = np.unique(labels)
    idx = [np.where(labels == i)[0] for i in numClassesList]

    for idxA in range(len(sentences)):
        currentSentence = sentences[idxA]
        label = labels[idxA]
        idxB = np.random.choice(idx[np.where(numClassesList == label)[0][0]])
        posSentence = sentences[idxB]
        # prepare a positive pair and update the sentences and labels
        # lists, respectively
        pairs.append(InputExample(texts=[currentSentence, posSentence], label=1.0))

        negIdx = np.where(labels != label)[0]
        negSentence = sentences[np.random.choice(negIdx)]
        # prepare a negative pair of images and update our lists
        pairs.append(InputExample(texts=[currentSentence, negSentence], label=0.0))

    # return a 2-tuple of our image pairs and labels
    return pairs


class SetFitClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        model: str,
        classifier_head=LogisticRegression(),
        loss=losses.CosineSimilarityLoss,
        random_state: int = 1234,
    ):
        random.seed(random_state)
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        self.model = SentenceTransformer(model)
        self.classifier = classifier_head
        self.loss = loss(self.model)
        self.fitted = False

    def fit(self, X, y, data_iter: int = 5, train_iter: int = 1):
        # TODO: Fix this (mutating + state issues)
        train_examples = []
        for x in range(data_iter):
            train_examples = sentence_pairs_generation(
                np.array(X), np.array(y), train_examples
            )
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
        self.model.fit(
            train_objectives=[(train_dataloader, self.loss)],
            epochs=train_iter,
            warmup_steps=10,
            show_progress_bar=True,
        )

        X_train = self.model.encode(X)
        self.classifier.fit(X_train, y)
        self.fitted = True

    def predict(self, X, y=None):
        if not self.fitted:
            raise NotFittedError(
                "This SetFitClassifier instance is not fitted yet."
                " Call 'fit' with appropriate arguments before using this estimator."
            )
        X_embed = self.model.encode(X)
        preds = self.classifier.predict(X_embed)
        return preds

    def predict_proba(self, X, y=None):
        if not self.fitted:
            raise NotFittedError(
                "This SetFitClassifier instance is not fitted yet."
                " Call 'fit' with appropriate arguments before using this estimator."
            )
        X_embed = self.model.encode(X)
        preds = self.classifier.predict_proba(X_embed)
        return preds
