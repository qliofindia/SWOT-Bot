"""Custom scikit-learn estimators used by the benchmark (and usable in production)."""

from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.linear_model import LogisticRegression


class NBSVM(ClassifierMixin, BaseEstimator):
    """Naive-Bayes-weighted logistic regression (Wang & Manning, 2012), one-vs-rest."""

    def __init__(self, C: float = 4.0, alpha: float = 1.0):
        self.C = C
        self.alpha = alpha

    def fit(self, X, y):
        X = sparse.csr_matrix(X)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.r_, self.models_ = [], []
        for c in self.classes_:
            pos = y == c
            p = self.alpha + X[pos].sum(axis=0)
            q = self.alpha + X[~pos].sum(axis=0)
            r = np.asarray(np.log((p / p.sum()) / (q / q.sum()))).ravel()
            self.r_.append(r)
            self.models_.append(
                LogisticRegression(C=self.C, max_iter=2000).fit(X.multiply(r).tocsr(), pos)
            )
        return self

    def predict_proba(self, X):
        X = sparse.csr_matrix(X)
        p = np.column_stack(
            [m.predict_proba(X.multiply(r).tocsr())[:, 1] for m, r in zip(self.models_, self.r_)]
        )
        return p / p.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


_SPACY = {}


class SpacyVectors(TransformerMixin, BaseEstimator):
    """Mean of spaCy en_core_web_md word vectors (300-d). Stateless; cached per text."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if "nlp" not in _SPACY:
            import spacy

            _SPACY["nlp"] = spacy.load("en_core_web_md", disable=["parser", "ner", "lemmatizer"])
            _SPACY["cache"] = {}
        nlp, cache = _SPACY["nlp"], _SPACY["cache"]
        todo = [t for t in dict.fromkeys(X) if t not in cache]
        for t, doc in zip(todo, nlp.pipe(todo, batch_size=64)):
            cache[t] = doc.vector
        return np.vstack([cache[t] for t in X])

    def __getstate__(self):
        return {}
