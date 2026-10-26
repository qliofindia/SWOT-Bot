from pathlib import Path

import pandas as pd
import pytest

from swotbot.config import settings
from swotbot.model import LABELS, SwotClassifier, load_dataset


def test_dataset_is_single_label():
    texts, labels = load_dataset(settings.data_path)
    assert len(texts) == len(labels) > 0
    assert set(labels) == set(LABELS)


def test_rejects_multi_label_rows(tmp_path: Path):
    bad = tmp_path / "bad.csv"
    pd.DataFrame([{"Text": "x", "STRENGTHS": 1, "WEAKNESSES": 1, "OPPORTUNITIES": 0, "THREATS": 0}]).to_csv(bad)
    with pytest.raises(ValueError):
        load_dataset(bad)


def test_save_load_roundtrip(tmp_path: Path):
    model = SwotClassifier.train(settings.data_path, cv_folds=0)
    model.save(tmp_path)
    loaded = SwotClassifier.load_or_train(tmp_path, settings.data_path)
    text = ["Increasing thrust on improving efficiencies to drive growth."]
    assert loaded.predict(text) == model.predict(text)
    assert model.predict([]) == []
