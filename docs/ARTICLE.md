# SWOT-Bot: turning corporate documents into a SWOT analysis with NLP

## What a corporate SWOT analysis is

A SWOT analysis summarises a company's position under four headings. Two are
internal, things the company controls. Two are external, conditions in its
market and environment.

- **Strengths** are internal advantages: a strong parent group, low debt, a
  loyal customer base, efficient working-capital management.
- **Weaknesses** are internal limitations: dependence on a few customers, a long
  collection period, thin margins, high staff attrition.
- **Opportunities** are external conditions the company can use: a growing
  market, new geographies, a favourable policy change.
- **Threats** are external conditions that can hurt it: an industry slowdown,
  currency volatility, new competitors, regulation.

The internal/external split matters because it decides what can be done. A
weakness can be fixed by management. A threat can only be monitored, hedged or
avoided.

SWOT analyses are used wherever someone must form a view of a company quickly
and defend it: credit appraisal and rating, investment research, due diligence
before an acquisition, supplier and counterparty risk reviews, and strategy
reviews inside the company.

## Why it is slow to do by hand

The evidence for a SWOT is spread across many documents: annual reports,
business and financial highlights, executive summaries, audit and risk reports,
industry overviews, legal documents, policies and procedures, press releases,
media coverage and company announcements. For one company this can run to
hundreds of pages.

An analyst reads all of it, picks out the passages that say something material,
decides which quadrant each belongs to, and writes them up. The reading and
sorting take most of the time. The judgement that matters, what the SWOT means
for a lending decision or an investment thesis, comes at the end, often under
time pressure.

Doing this by hand has three further costs. Two analysts sort the same passage
differently, so results vary between people. Material in long documents gets
missed. And the finished report rarely records which source sentence each point
came from, which makes review and audit harder.

## What SWOT-Bot does

SWOT-Bot automates the sorting step. It takes passages of narrative text,
assigns each one to Strengths, Weaknesses, Opportunities or Threats, attaches a
confidence score, and returns the results so they can be consolidated into a
Word document titled "SWOT Analysis."

Here are three passages from the training data and how they are classified:

> "The Company undertook significant capital expenditure... funded through
> internal accruals. Low gearing gives financial flexibility to the Company to
> raise additional funds for future capex plans." (Strength: internal, positive)

> "Collection period of the Group remained high and ranged between 113 and 128
> days... significantly high when compared with the management's credit policy
> of 30–60 days." (Weakness: internal, negative)

> "The group has forayed into multiple new countries including Netherland,
> Singapore and Dubai through wholly owned subsidiaries." (Opportunity:
> external, positive)

The analyst's job changes from reading and sorting everything to reviewing a
pre-sorted draft. They correct the misclassified passages, which are flagged by
low confidence, and spend their time on interpretation.

## Where it adds value

These are the expected benefits. The project has not yet run a time-and-motion
study, so treat them as hypotheses to measure in a pilot, not as results.

- **Throughput.** Classifying a passage takes milliseconds, so reading hundreds
  of pages becomes reviewing a sorted list. The gain is largest when the same
  kind of analysis is repeated across many companies, as in a credit or
  research team.
- **Consistency.** The same passage always gets the same label, which reduces
  differences between analysts and between reviews of the same company over
  time.
- **Coverage.** Every passage fed in is classified, so material in long
  documents is less likely to be missed.
- **Traceability.** Each point in the report stays linked to its source
  passage, so a reviewer or auditor can check where a claim came from.
- **Triage.** Confidence scores show which passages need a person's attention,
  so review effort goes where the model is unsure.

What it does not do is just as important. It does not judge how much a strength
or threat matters, it does not check whether a claim in a document is true, and
it does not replace the analyst's conclusion. It is a first-draft generator with
a human in the loop.

## How the system fits together

There are five stages:

1. **Collect.** Gather the documents for one company and extract their
   narrative text.
2. **Segment.** Split the text into passages: paragraphs or bullet points, each
   making one point.
3. **Classify.** Send the passages to the model, which returns a label and the
   probability of each class.
4. **Review.** An analyst checks low-confidence passages and corrects labels.
5. **Consolidate.** Group the passages by label and write them into the four
   sections of the Word report.

The model runs as a small HTTP service, so any tool that can send an HTTP
request (a script, a document pipeline or an internal application) can
classify passages.

## Building it, step by step

### Step 1: prepare labelled training data

A supervised model learns from examples labelled by people. The training file,
`data/train.csv`, holds 2,500 passages from Indian corporate credit and
business-analysis reports. Each row has the passage text and four 0/1 columns,
one per quadrant, with exactly one set to 1:

```
#id,Text,Label,STRENGTHS,WEAKNESSES,OPPORTUNITIES,THREATS
1,Long-standing presence in industry coupled with part of TVS group,1,1,0,0,0
```

The classes are moderately balanced: 827 strengths, 619 weaknesses, 549
opportunities and 505 threats. Passages are a median of 93 tokens long, and
the longest is 467. Those lengths matter in step 3.

The labels must be clean. A smaller file, `data/test.csv`, turned out to be
mislabelled; for example, a passage about a "competitive edge over peers" is
marked as a weakness. It is excluded from evaluation. One wrong label in a
16-row test set moves accuracy by six points.

### Step 2: choose the tooling

The first version of this project used ktrain, a convenience wrapper around
TensorFlow Keras, to fine-tune BERT in a few lines. ktrain's last release was
in June 2024, and its BERT path downloads weights from a Google storage bucket
that now refuses anonymous downloads. The rebuilt project uses Hugging Face
`transformers` and PyTorch for BERT, and scikit-learn for the classical models.
All are actively maintained.

The steps ktrain bundled together still exist, done explicitly: tokenising
passages with BERT's tokenizer (which adds the `[CLS]` and `[SEP]` markers),
adding a four-way classification layer, and training with a learning-rate
schedule that warms up and then decays.

### Step 3: fine-tune BERT

```python
from swotbot.model import load_dataset
from benchmarks.neural import HFFineTuneClassifier

texts, labels = load_dataset("data/train.csv")
bert = HFFineTuneClassifier("bert-base-uncased", max_len=256, epochs=3, lr=3e-5, batch_size=16)
bert.fit(texts, labels)
bert.predict_proba(["Manpower attrition risk along with increasing salary cost."])
```

The most important parameter is `max_len`, the number of tokens the model reads
from each passage. The first version used 32. Since the median passage is 93
tokens, 74 percent of passages were cut off and the model never saw most of
their text. At 256 tokens only 3 percent are truncated.

Hardware decides how practical this is. On a 4-core CPU, one epoch over 2,000
passages took about 24 minutes. On one NVIDIA A10G GPU, all three epochs took
37 seconds. That is why the heavy training runs on Hugging Face (see "Running
the training on Hugging Face" below).

### Step 4: measure it honestly

The first version reported 95 percent accuracy "upon training." That number was
measured on the passages the model had just been trained on, so it shows how
well the model memorised them. It is not evidence of how the model handles a new
annual report.

The rebuilt project uses five-fold cross-validation. The data is split into
five parts; the model is trained on four and tested on the fifth, and this
repeats until each part has been the test set once. Every passage is scored by
a model that never saw it.

The headline metric is macro-F1, the average of the F1 score for each class.
Plain accuracy can look good while the model neglects a smaller class. Macro-F1
counts weaknesses and threats as much as strengths.

### Step 5: compare BERT against the alternatives

BERT is not automatically the best choice. It is large, slow on ordinary
hardware, and with only 2,500 examples it may not beat simpler methods. So 28
models were benchmarked on the same five folds. They fall into these groups:

- classical classifiers on word and phrase frequencies (logistic regression,
  support vector machines, naive Bayes, gradient-boosted trees);
- word-embedding methods (spaCy vectors, GloVe and Doc2Vec via Gensim);
- fastText;
- neural networks trained on this data (a feed-forward network, a
  convolutional network, a bidirectional LSTM and a small Transformer);
- pretrained sentence encoders;
- BERT, DistilBERT, FinBERT (BERT further trained on financial text) and
  RoBERTa.

Each model was compared with the leader using a statistical test that accounts
for the overlap between cross-validation training sets. A model is only called
better when the gap is larger than fold-to-fold noise.

### What the benchmark found

Twenty-seven of the 28 models completed. FinBERT failed on the first run: its
published checkpoint carries a three-class sentiment head, and the code did not
allow that head to be replaced with a four-class one. The fix is a one-line
change, and FinBERT will be rerun on its own.

**Fine-tuned transformers came first.** RoBERTa-base reached a macro-F1 of 0.875
(accuracy 0.879), BERT-base 0.867 and DistilBERT 0.863. These three are
statistically indistinguishable from one another: RoBERTa against BERT gives
p = 0.54, and against DistilBERT p = 0.24. RoBERTa also produced the most
reliable probabilities, with the lowest log loss of any model (0.392).

**The best classical models came close.** A soft-voting ensemble of three linear
models scored 0.853. A single ridge classifier on TF-IDF features scored 0.850.
Logistic regression, the model the service currently uses, scored 0.845.

**How large is the transformers' lead?** RoBERTa beats the current production
model by three points, and that gap is statistically significant (p = 0.04). It
beats the ensemble by 2.2 points, which only just clears the bar (p = 0.049).
Against ridge (2.5 points, p = 0.07) and word-and-character logistic regression
(2.8 points, p = 0.13) the gap does not reach significance. With 2,500 passages
and five folds, the honest reading is that fine-tuned transformers are probably
better by two to three points, but this dataset is too small to prove it beyond
doubt.

**Neural networks trained from scratch did not help.** A feed-forward network on
TF-IDF features matched the linear models at 0.847, because it learns the same
phrase signals. The convolutional network (0.829), the bidirectional LSTM
(0.810) and the small Transformer (0.786) all scored lower. Learning language
patterns from 2,500 examples is too much to ask of these architectures; models
pretrained on large corpora start with that knowledge.

**Averaged embeddings did worst.** Frozen sentence encoders scored between
0.775 and 0.819. Averaged word vectors (GloVe 0.774, Doc2Vec 0.779, spaCy
0.733) and fastText (0.771 from scratch, 0.817 with GloVe initialisation)
trailed the simple TF-IDF models by three to eleven points.

The pattern makes sense for this task. A SWOT label depends on two things:
direction (good or bad) and locus (inside or outside the company). Both are
carried by specific phrases, such as "collection period remained high," "low
gearing" or "slowdown in the automobile industry." TF-IDF keeps those phrases
intact. Averaging word vectors blurs them into a general topic, which is why
embedding models confuse a strong balance sheet with a weak one. Fine-tuned
transformers read the phrases in context, including words like "however" and
"despite," which is where their extra points come from.

**Accuracy has a cost.** The fine-tuned RoBERTa model is 475 MB. Each fold took
38 seconds to train on the GPU, and it classifies a passage in 1.7 ms on the
GPU. On a CPU it is much slower; measure it on the target hardware before
deploying. The ridge classifier is 835 KB, trains in under a second, and
classifies a passage in 0.09 ms on an ordinary CPU.

**Compared with the original project:** the first version claimed 95 percent
from training accuracy. Measured properly with cross-validation, BERT reaches
87 percent accuracy on this data, and only after the input length is raised
from 32 to 256 tokens.

**Recommendation.** Where accuracy matters most and a GPU, or a few
milliseconds of extra latency per passage, is acceptable, use fine-tuned
RoBERTa-base. Keep the TF-IDF ensemble as a lightweight fallback: it needs no
GPU, has no download, and gives up only about two points. Either way, the
largest remaining gain is more and cleaner labelled data, not a different model.

The full results for every model, including per-fold scores, are in
`benchmarks/RESULTS.md`. To rerun any model, open
`notebooks/benchmark.ipynb`, which has one cell per model.

### Step 6: train and save the production model

```bash
pip install -e ".[dev]"
python -m swotbot train
```

The service currently ships the TF-IDF logistic-regression model. It needs no
GPU, which makes it the easiest to run anywhere. Serving the fine-tuned RoBERTa
model behind the same API is the next engineering step.

This command runs cross-validation, prints the scores, trains on all 2,500
passages and saves the model with its metadata: the library version, a fingerprint of the
training data, and the scores. On startup the service checks this metadata and
retrains automatically if the model is missing or out of date.

### Step 7: serve it

```bash
python -m swotbot serve        # http://127.0.0.1:8088, interactive docs at /docs
```

The service replaces the original Flask and gevent setup with FastAPI. It
accepts batches of up to 256 passages per request and checks input sizes.
`/health` reports which model is loaded and how it scored.

### Step 8: classify passages

From Python, send a batch and get back a label, a confidence and all four class
probabilities for each passage:

```python
import requests

passages = [
    "Long-standing presence in industry coupled with strong group support.",
    "Exposure to foreign exchange fluctuation risk due to 99% export revenue.",
]
r = requests.post("http://127.0.0.1:8088/predict", json={"texts": passages})
for item in r.json():
    print(item["label"], round(item["confidence"], 2), item["text"][:60])
```

### Step 9: consolidate into the report

Group the classified passages by label, sort each group by confidence, and
write the four groups into the Strengths, Weaknesses, Opportunities and Threats
sections of the Word template. Keep the source document and page with each
passage so the report stays traceable. Passages below a confidence threshold,
for example 0.5, go into a "for review" list instead of the report body.

## Running the training on Hugging Face

The project was rebuilt with Claude Code working in a cloud container with
4 CPU cores and no GPU. Getting BERT-scale training to work took these steps.

**Open the network.** The container could not reach `huggingface.co` at first,
so pretrained models could not be downloaded. The fix was a setting, not code:
add `huggingface.co` and `*.hf.co` to the environment's allowed domains, then
start a new session.

**Handle the token safely.** The Hugging Face token lives in a `.env` file that
git ignores. Claude scanned every commit for anything shaped like a key, and
checked the token with `HfApi().whoami()` without printing it. Two routes failed
and should be skipped. The AWS-style keys already in the environment were
placeholders, so the S3 route to the bucket failed. `hf-mount` needs FUSE
support, which containers often lack. The `huggingface_hub` library reads and
writes buckets directly.

**Use Jobs for compute.** A Hugging Face bucket only stores files. Hugging Face
Jobs runs code on rented hardware and can mount a bucket as a folder. The
benchmark ran on an `a10g-small` machine (one NVIDIA A10G GPU, $1.00 per hour
when this was written) with a hard two-hour timeout, which capped the cost at
$2. The full run of 27 models took 44.5 minutes, about $0.74 at that rate.

**Package the run.** A launcher copies an explicit list of files (the code and
the training data, never `.env`) into the bucket and starts the job. The job
installs pinned dependencies, runs the benchmark, and writes each model's
results to the bucket as soon as that model finishes. The bucket is mounted at
`/bucket`, because `/data` is reserved by Jobs.

**Make failures cheap.** An error in one model is logged and the run moves on.
A hard crash (out of memory, GPU fault) kills the process, as it would kill a
notebook kernel. Even then, finished models are already saved, and relaunching
skips them and resumes at the first unfinished one. The same logic is available
as a notebook, `notebooks/benchmark.ipynb`, with one cell per model. A failed
cell prints its error and the next cell still runs, so a single model can be
fixed and rerun on its own.

## Limitations and controls

The most common errors are confusing strengths with weaknesses, which use the
same vocabulary with opposite meaning, and weaknesses with threats, which differ
mainly in whether the risk is internal or external. An analyst should review
the draft before it is used in a decision.

The training data comes from one genre, Indian credit-rating narratives.
Accuracy on press releases, legal documents or media coverage is likely to be
lower until labelled examples from those sources are added. A pilot should
measure accuracy on each document type before relying on it.

The confidence score is a model estimate, not a guarantee. It is useful for
sorting work, but it should be checked against real review outcomes to see
whether, say, 80 percent confidence really means about 80 percent correct.

## What comes next

The largest gains are likely to come from data. Labelling passages from the
other document types, and building an independent test set that is never used
for tuning, matter more than switching models. On the engineering side, the
next steps are to serve the fine-tuned RoBERTa model behind the existing API,
and to generate the Word report on the server, so the full pipeline from
documents to "SWOT Analysis" runs as one command.
