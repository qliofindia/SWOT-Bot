# SWOT-Bot

Classifies a sentence or paragraph from a business or credit analysis as one of
**STRENGTHS, WEAKNESSES, OPPORTUNITIES, THREATS**, and serves the result over
HTTP so the Word add-in in `word-addin/` can call it.

Version 2 replaces the 2020 stack (ktrain, TensorFlow 1-era BERT code, Flask +
gevent) with scikit-learn and FastAPI on Python 3.11+.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

python -m swotbot train          # ~5 s; writes models/ and prints CV metrics
python -m swotbot serve          # http://127.0.0.1:8088  (docs at /docs)
python clients/request_example.py
pytest
```

If `models/` is missing, was built with a different scikit-learn version, or
`data/train.csv` has changed, the server retrains on startup automatically.

## API

| Method | Path | Body / query | Response |
|---|---|---|---|
| GET | `/health` | | status, classes, model metadata (CV scores) |
| POST | `/predict` | `{"texts": ["...", "..."]}` (1–256 texts) | `[{text, label, confidence, scores}]` |
| POST | `/api/?TEXT=...` | legacy, used by `word-addin/httpRequest.bas` | `["THREATS", 52.22]` (confidence in %) |

`/api/` keeps the original contract, including the `['...']` wrapping the VBA
macro puts around the text.

Configuration is through environment variables or `.env`: `SWOTBOT_DATA_PATH`,
`SWOTBOT_MODEL_DIR`, `SWOTBOT_MAX_TEXT_CHARS`, `SWOTBOT_MAX_BATCH`.

## Model

TF-IDF (word 1–2-grams, sublinear tf) followed by multinomial logistic regression.
I compared it under 5-fold stratified cross-validation on `data/train.csv`
(2,500 rows) against word+char n-grams, LinearSVC, calibrated SVC and
ComplementNB. All were within ±0.01 accuracy of each other (≈0.85), so I
picked the simplest model that gives native probabilities.

Cross-validated results:

| Class | Precision | Recall | F1 |
|---|---|---|---|
| STRENGTHS | 0.855 | 0.880 | 0.868 |
| WEAKNESSES | 0.794 | 0.761 | 0.777 |
| OPPORTUNITIES | 0.882 | 0.902 | 0.892 |
| THREATS | 0.851 | 0.834 | 0.842 |
| **Accuracy** | | | **0.846** |

The main confusions are STRENGTHS↔WEAKNESSES (opposite polarity, same
vocabulary) and WEAKNESSES↔THREATS (internal versus external risk). A bag of
n-grams cannot model negation or context. A fine-tuned transformer is the
obvious next step if more accuracy is needed (see below).

### Known data issues

- `data/test.csv` (16 rows) appears to be mislabelled. For example, row 2501
  describes a competitive edge but is labelled WEAKNESSES, and the file has no
  STRENGTHS rows. Do not use it for evaluation; the metrics above come from
  cross-validation on `train.csv`.
- The CSVs are Latin-1, and apostrophes were lost as `?` (for example `debtor?s`).

## Layout

```
swotbot/        package: model.py (train/load/predict), api.py (FastAPI), config.py, CLI
tests/          pytest suite (API contract, data validation, save/load)
clients/        example HTTP client
data/           training data (CSV + original Excel copies)
word-addin/     Word templates with the VBA client; httpRequest.bas extracted for reading
notebooks/      original ktrain / BERT / MongoDB notebooks (historical, not maintained)
legacy/         original v1 Flask apps, request scripts and Google BERT TF1 code, unmodified
docs/demo.mp4   original demo video (was `videoplayback`)
```

## Upgrading to a transformer later

The original project fine-tuned BERT with `maxlen=32`, which truncated most
paragraphs in the data. Its trained weights were never committed (only
`tf_model.preproc`), so it can't be reproduced or compared. To try a
transformer, fine-tune a model such as `deberta-v3-base` with Hugging Face
`transformers` using `max_length` ≥ 256. Wrap it in a class with the same
`predict(texts) -> list[Prediction]` method, and compare it against the
0.846 CV baseline on the same folds before switching.
