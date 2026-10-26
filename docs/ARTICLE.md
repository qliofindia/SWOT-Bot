# SWOT-Bot: classifying corporate text into Strengths, Weaknesses, Opportunities and Threats

## The problem

A company produces a lot of narrative text: annual reports, business and
financial highlights, executive summaries, credit and audit reports, risk
reports, industry overviews, policies, press releases and blog posts. An analyst
building a SWOT analysis reads all of it and sorts each relevant passage into one
of four buckets:

- **Strength:** an internal advantage ("low gearing gives financial flexibility")
- **Weakness:** an internal limitation ("collection period of 113–128 days against a 30–60 day policy")
- **Opportunity:** an external factor the firm can use ("foray into Netherlands, Singapore and Dubai")
- **Threat:** an external risk ("slowdown in the automobile sector, the main customer industry")

SWOT-Bot automates the sorting step. It takes passages, labels each one S, W, O
or T with a confidence score, and returns the labelled passages so they can be
consolidated into a Word document titled "SWOT Analysis".

This is single-label, four-class text classification. Each passage belongs to
exactly one quadrant.

## The pipeline

1. **Collect.** Extract narrative passages (paragraphs or bullet points) from the source documents.
2. **Classify.** Send the passages to the model, which returns a label and class probabilities for each.
3. **Consolidate.** Group the passages by label and write them into the four sections of a Word report.

The Word add-in in `word-addin/` covers steps 2 and 3 from inside Word. It sends
the selected text to a local HTTP service and shows the result.

## The data

`data/train.csv` contains 2,500 labelled passages from Indian corporate credit
and business-analysis reports. Each row has the text and a one-hot label:

| #id | Text | STRENGTHS | WEAKNESSES | OPPORTUNITIES | THREATS |
|---|---|---|---|---|---|
| 1 | Long-standing presence in industry coupled with part of TVS group | 1 | 0 | 0 | 0 |

Class counts: S 827, W 619, O 549, T 505. Passages range from one line to a few
hundred words. The median is 93 BERT tokens, the 90th percentile 204, and the
maximum 467.

Two data issues matter:

- `data/test.csv` (16 rows) is mislabelled. For example, a passage about
  "competitive edge over peers" is labelled WEAKNESS, and there are no STRENGTHS
  rows at all. It is not used for evaluation.
- Apostrophes were lost in an encoding step (`debtor?s`). This is harmless for
  the models but worth fixing at the source.

## Choosing a model

The first version used ktrain to fine-tune BERT. ktrain is a Keras wrapper whose
last release was in June 2024. The original notebook set `MAXLEN = 32`, so 74% of
passages were cut off after 32 tokens. Its reported 95% "accuracy upon training"
was training-set accuracy, which shows the model memorised the data, not that it
generalises. The trained weights were never committed, so that result cannot be
reproduced.

Version 2 does not assume BERT is the answer. It benchmarks 28 models from
nine families on the same data with the same protocol:

| Family | Models |
|---|---|
| Linear on TF-IDF | logistic regression, linear SVM, SGD, ridge, LSA + logistic regression |
| Probabilistic | complement naive Bayes, NB-SVM |
| Tree ensemble | LightGBM on LSA features |
| Ensemble | soft vote of the three best linear models |
| Static word embeddings | spaCy vectors, Gensim GloVe-300, Gensim Doc2Vec |
| fastText | trained from scratch; initialised from GloVe |
| Neural, trained here | MLP (ANN) on TF-IDF, Kim CNN, BiLSTM with attention, 2-layer Transformer |
| Pretrained transformer, frozen | MiniLM, MPNet, BGE sentence embeddings + logistic regression |
| Pretrained transformer, fine-tuned | DistilBERT, BERT-base, FinBERT, RoBERTa-base (max length 256) |

**Protocol.** Five-fold stratified cross-validation with a fixed seed, using
identical folds for every model. Neural models use 10% of each training fold for
early stopping; the test fold is never seen during training. The main metric is
macro-F1, the unweighted mean F1 over the four classes, so the smaller classes
count as much as STRENGTHS. Probability quality is measured with log loss.

**Deciding whether a difference is real.** With 2,500 examples, a 0.5-point gap
between two models can be noise. Each model is compared with the top one using
the Nadeau–Bengio corrected resampled t-test. A plain paired t-test over CV
folds understates the variance because the training sets overlap. A model is
only called better if p < 0.05.

The benchmark runs as a Hugging Face Job on a single A10G GPU and writes results
to `benchmarks/RESULTS.md`:

```bash
python benchmarks/launch_hf_job.py            # upload code, start the GPU job
python benchmarks/launch_hf_job.py --fetch    # pull results
python benchmarks/benchmark.py report         # rebuild the table
```

### Results

<!-- RESULTS -->

## Training the production model

```bash
pip install -e ".[dev]"
python -m swotbot train
```

`train` runs 5-fold cross-validation, prints accuracy and macro-F1, fits the
model on all 2,500 rows, and saves it to `models/` together with its metadata.
The metadata records the scikit-learn version, a hash of the training data and
the CV scores. On startup the service retrains automatically if the model is
missing, was built with another library version, or the data has changed.

## Serving

```bash
python -m swotbot serve          # http://127.0.0.1:8088, interactive docs at /docs
```

| Endpoint | Purpose |
|---|---|
| `POST /predict` with `{"texts": [...]}` | batch classification; returns label, confidence and all four class probabilities per passage |
| `POST /api/?TEXT=...` | the original contract used by the Word add-in; returns `["THREATS", 52.22]` |
| `GET /health` | status, class list and the model's CV scores |

```python
import requests
r = requests.post("http://127.0.0.1:8088/predict",
                  json={"texts": ["Manpower attrition risk along with increasing salary cost."]})
print(r.json())
```

## Limitations

- About 15% of passages are misclassified, mostly STRENGTH↔WEAKNESS (same
  vocabulary, opposite direction) and WEAKNESS↔THREAT (internal versus external
  risk). Treat the output as a first draft for an analyst, not a final report.
- The training data comes from one style of report: Indian credit-rating
  narratives. Expect lower accuracy on press releases, blogs or legal text
  until examples from those sources are labelled and added.
- Confidence is a model probability, not a guarantee. Low-confidence passages
  (for example under 0.5) should be routed to a person.

## What would improve it most

1. **More, cleaner labels.** Relabel `test.csv` and add passages from the other
   document types. Data quality is the main limit at 2,500 examples.
2. **A proper held-out test set,** labelled independently and never used for
   model selection.
3. **Report generation.** Turn the classified passages into the Word "SWOT
   Analysis" document server-side (for example with `python-docx`), so the
   pipeline works without the VBA add-in.
