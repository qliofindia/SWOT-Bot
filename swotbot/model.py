"""SWOT sentence classifier: training, persistence and inference."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

log = logging.getLogger(__name__)

LABELS = ["STRENGTHS", "WEAKNESSES", "OPPORTUNITIES", "THREATS"]
MODEL_FILE = "swot_model.joblib"
META_FILE = "swot_model.json"


def build_pipeline() -> Pipeline:
    # Chosen by 5-fold stratified CV on data/train.csv: word+char n-grams,
    # LinearSVC and calibrated SVC all scored within noise (~0.85 acc) of this,
    # so the simplest model with native probabilities wins.
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                    min_df=2,
                    strip_accents="unicode",
                ),
            ),
            ("clf", LogisticRegression(C=10, max_iter=2000)),
        ]
    )


def load_dataset(path: Path) -> tuple[pd.Series, pd.Series]:
    """Read the one-hot SWOT CSV and return (texts, label names)."""
    df = pd.read_csv(path, encoding="latin1")
    missing = {"Text", *LABELS} - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    df = df.dropna(subset=["Text"])
    onehot = df[LABELS]
    if not (onehot.sum(axis=1) == 1).all():
        raise ValueError(f"{path}: every row must have exactly one SWOT label set")
    return df["Text"].astype(str), onehot.idxmax(axis=1)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float
    scores: dict[str, float]


class SwotClassifier:
    def __init__(self, pipeline: Pipeline, meta: dict):
        self.pipeline = pipeline
        self.meta = meta
        self.classes: list[str] = list(pipeline.classes_)

    @classmethod
    def train(cls, data_path: Path, cv_folds: int = 5) -> "SwotClassifier":
        texts, labels = load_dataset(data_path)
        meta: dict = {
            "data_sha256": _sha256(data_path),
            "n_samples": len(texts),
            "sklearn_version": sklearn.__version__,
        }
        if cv_folds > 1:
            cv = StratifiedKFold(cv_folds, shuffle=True, random_state=42)
            r = cross_validate(
                build_pipeline(), texts, labels, cv=cv, scoring=["accuracy", "f1_macro"]
            )
            meta["cv"] = {
                "folds": cv_folds,
                "accuracy": round(float(r["test_accuracy"].mean()), 4),
                "accuracy_std": round(float(r["test_accuracy"].std()), 4),
                "f1_macro": round(float(r["test_f1_macro"].mean()), 4),
            }
            log.info("cross-validation: %s", meta["cv"])
        pipeline = build_pipeline().fit(texts, labels)
        return cls(pipeline, meta)

    def save(self, model_dir: Path) -> None:
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, model_dir / MODEL_FILE)
        (model_dir / META_FILE).write_text(json.dumps(self.meta, indent=2))

    @classmethod
    def load(cls, model_dir: Path) -> "SwotClassifier":
        meta = json.loads((model_dir / META_FILE).read_text())
        return cls(joblib.load(model_dir / MODEL_FILE), meta)

    @classmethod
    def load_or_train(cls, model_dir: Path, data_path: Path) -> "SwotClassifier":
        """Load a saved model; retrain if absent, stale, or from another sklearn."""
        try:
            model = cls.load(model_dir)
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            log.info("no usable saved model (%s); training", exc)
        else:
            if model.meta.get("sklearn_version") != sklearn.__version__:
                log.info("saved model built with another scikit-learn; retraining")
            elif data_path.exists() and model.meta.get("data_sha256") != _sha256(data_path):
                log.info("training data changed; retraining")
            else:
                return model
        model = cls.train(data_path)
        model.save(model_dir)
        return model

    def predict(self, texts: list[str]) -> list[Prediction]:
        if not texts:
            return []
        proba = self.pipeline.predict_proba(texts)
        best = proba.argmax(axis=1)
        return [
            Prediction(
                label=self.classes[i],
                confidence=float(p[i]),
                scores={c: round(float(s), 4) for c, s in zip(self.classes, p)},
            )
            for i, p in zip(best, np.asarray(proba))
        ]
