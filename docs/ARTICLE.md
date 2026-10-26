# SWOT-Bot: classifying corporate text into Strengths, Weaknesses, Opportunities and Threats

## The problem

Companies produce a steady stream of narrative text: annual reports, business
and financial highlights, executive summaries, credit and audit reports, risk
reports, industry overviews, policies, announcements and blog posts. An analyst
preparing a SWOT analysis reads this material and sorts each relevant passage
into one of four groups.

A **strength** is an internal advantage, for example "low gearing gives
financial flexibility." A **weakness** is an internal limitation, such as a
collection period of 113 to 128 days against a credit policy of 30 to 60 days.
An **opportunity** is an external factor the firm can use, like expansion into
new markets through subsidiaries. A **threat** is an external risk, for
instance a slowdown in the industry that buys most of the firm's output.

SWOT-Bot automates the sorting. It takes passages of text, assigns each one to
a quadrant with a confidence score, and returns the results so they can be
consolidated into a Word document titled "SWOT Analysis." In machine-learning
terms this is single-label text classification with four classes: each passage
belongs to exactly one quadrant.

## How the pipeline works

There are three stages. First, narrative passages (paragraphs or bullet points)
are extracted from the source documents. Second, each passage is sent to the
classifier, which returns a label and the probability of each of the four
classes. Third, the passages are grouped by label and written into the four
sections of the report.

The Word add-in in `word-addin/` handles the second and third stages from
inside Word. It sends the selected text to a local HTTP service and displays
the classification.

## The data

The training set, `data/train.csv`, contains 2,500 labelled passages taken from
Indian corporate credit and business-analysis reports. Each row holds the text
and a one-hot label across the four columns STRENGTHS, WEAKNESSES,
OPPORTUNITIES and THREATS. The classes are moderately imbalanced: 827
strengths, 619 weaknesses, 549 opportunities and 505 threats.

Passages vary from a single line to a few hundred words. Measured in BERT
tokens, the median passage is 93 tokens long, the 90th percentile is 204, and
the longest is 467.

Two problems in the data affect how results should be read. The small file
`data/test.csv` (16 rows) is mislabelled: a passage describing a "competitive
edge over peers" is marked as a weakness, and the file contains no strengths at
all. It is therefore not used for evaluation. Separately, apostrophes were lost
in an earlier encoding step, so "debtor's" appears as "debtor?s." This does not
affect the models much, but it should be fixed at the source.

## Why the first version was replaced

The first version fine-tuned BERT through ktrain, a Keras wrapper whose last
release was in June 2024. Three issues made it unreliable.

The notebook set the maximum sequence length to 32 tokens. Because the median
passage is 93 tokens, 74 percent of passages were truncated, and the model never
saw most of the text it was meant to classify.

The headline figure of 95 percent accuracy was measured on the training data.
Training accuracy shows how well a model fits examples it has already seen. It
says little about how it performs on new passages.

The trained weights were never committed to the repository, so the result
cannot be reproduced or checked.

Version 2 does not assume that BERT, or any other model, is the right choice. It
runs a controlled comparison and lets the measurements decide.

## How the models were compared

Twenty-eight models from nine families were evaluated on the same data under
the same protocol.

The classical baselines are linear models on TF-IDF features: logistic
regression, linear support vector machines, stochastic gradient descent, ridge
regression, and logistic regression on a 300-dimensional latent semantic
analysis (LSA) projection. They are joined by two probabilistic models
(complement naive Bayes and NB-SVM), a gradient-boosted tree model (LightGBM on
LSA features), and a soft-voting ensemble of the strongest linear models.

The word-embedding models represent each passage by averaging pretrained word
vectors from spaCy or from GloVe (loaded through Gensim), or by learning
paragraph vectors with Gensim's Doc2Vec. fastText was trained both from scratch
and with GloVe initialisation.

The neural networks trained on this data are a feed-forward network (MLP) over
TF-IDF features; a convolutional network in the style of Kim (2014), with
filter widths of three, four and five words; a bidirectional LSTM with
attention pooling; and a two-layer Transformer encoder trained from scratch.
The CNN, LSTM and Transformer start from GloVe word vectors.

The pretrained transformers were tested in two ways. In the frozen setting, a
sentence encoder (MiniLM, MPNet or BGE) turns each passage into a fixed vector
and logistic regression classifies it. In the fine-tuned setting, DistilBERT,
BERT-base, FinBERT (a BERT model pretrained on financial text) and RoBERTa-base
are trained end to end on the classification task, with a maximum length of 256
tokens. That length covers 97 percent of passages in full.

Every model used five-fold stratified cross-validation with a fixed random seed,
so all of them were trained and tested on the same splits. Neural models held
back 10 percent of each training fold for early stopping and never saw the
test fold during training. The main metric is macro-F1, the average F1 score
across the four classes, which weights the smaller classes equally with the
largest. Log loss is reported as a measure of how well calibrated the predicted
probabilities are.

With only 2,500 examples, a difference of half a point between two models can
be noise. Each model was therefore compared with the top-ranked model using the
Nadeau–Bengio corrected resampled t-test. This variant accounts for the overlap
between training sets in cross-validation; a plain paired t-test understates
the variance. A model counts as genuinely worse than the leader only when p is
below 0.05.

The full benchmark ran as a Hugging Face Job on a single NVIDIA A10G GPU. The
complete per-model results, including fit time, prediction latency and model
size, are in `benchmarks/RESULTS.md`, and they can be reproduced with:

```bash
python benchmarks/launch_hf_job.py            # upload code and start the GPU job
python benchmarks/launch_hf_job.py --fetch    # download the results
python benchmarks/benchmark.py report         # rebuild the report
```

## Benchmark results

<!-- RESULTS -->

## Using Hugging Face with Claude

The benchmark was built and run with Claude Code working in a cloud session, a
sandboxed Linux container with 4 CPU cores and no GPU. Getting BERT-class models
into that workflow took several steps. Some were dead ends, and they are worth
knowing about before you start.

**Step 1: allow the Hugging Face domains.** The container's network policy
blocked `huggingface.co` at first, so no pretrained model could be downloaded.
Package indexes (PyPI) and GitHub were reachable, so classical models, spaCy
vectors and GloVe (served from GitHub releases) worked, but BERT did not. The
fix is in the cloud environment settings, not in code: under Network access,
add `huggingface.co` and `*.hf.co` to the allowed domains and keep package
managers allowed. The second pattern matters because downloads, buckets and
the S3-compatible endpoint live on subdomains such as `s3.hf.co`. A change to
the network policy may only apply to sessions started after it, so plan on
opening a new session.

**Step 2: provide a token without committing it.** Claude stored the Hugging
Face access token in a `.env` file at the repository root. That file is listed
in `.gitignore`, along with patterns such as `*.key`, `*API_Key*` and
`secrets/`. Before each commit Claude scanned the staged changes for anything
shaped like a key. The token was checked with `HfApi().whoami()`, which
confirmed the account name and token type without printing the token. Two
things did not work and wasted time. The `AWS_ACCESS_KEY_ID` variables already
in the environment were placeholders, so the boto3 route to the bucket failed
with "InvalidAccessKeyId." The `hf-mount` route needs both network access and
FUSE support in the container. The `huggingface_hub` library (version 2.3) has
native bucket functions, so neither boto3 nor a mount is needed.

**Step 3: understand what a bucket is.** A Hugging Face bucket is storage. It
holds files but cannot run code. To run models on Hugging Face hardware you
need Hugging Face Jobs, which runs a container on a chosen machine. A job can
mount a bucket as a folder.

**Step 4: decide where the compute runs.** Claude timed one BERT-base training
step in the container before choosing. At 256 tokens, an epoch over 2,000
passages took about 24 minutes on 4 CPU cores, so five folds of three epochs
would have taken about six hours. On a Hugging Face `a10g-small` machine (one
A10G GPU, $1.00 per hour at the time of writing) the same work takes minutes.
The job was given a hard two-hour timeout, which caps the cost at $2 whatever
happens. Prices come from `HfApi().list_jobs_hardware()` and change, so check
them before running.

**Step 5: package the run.** The job runs a single `uv` script,
`benchmarks/hf_job.py`. Its pinned dependencies are declared in the script
header, so the job installs exactly what was tested. A launcher,
`benchmarks/launch_hf_job.py`, copies an explicit allow-list of files (the
code and `data/train.csv`, never `.env`) into the bucket under `code/`. It
then starts the job with the token passed as a job secret and the bucket
mounted. One detail matters: the mount path `/data` is reserved for job
artifacts when running local scripts, so the bucket is mounted at `/bucket`.

**Step 6: make failures cheap.** Each model's per-fold scores are written to
`/bucket/results/folds/` as soon as that model finishes. An ordinary Python
error inside one model is caught and logged, and the run moves on to the next
model. A hard crash, such as running out of memory or a GPU fault, kills the
process, just as it would kill a notebook kernel. Even then, every finished
model is already saved, and launching again skips completed models and resumes
at the first unfinished one. The notebook version of the benchmark follows the
same rule, one cell per model, so a failed model can be fixed and rerun alone.

**Step 7: monitor and collect.** Claude polled the job with
`HfApi().inspect_job()` and `fetch_job_logs()` and reported each model's score
as it finished. When the job ended, the results were pulled back into the
repository with `python benchmarks/launch_hf_job.py --fetch`, and the report
was rebuilt with `python benchmarks/benchmark.py report`.

## Training the production model

```bash
pip install -e ".[dev]"
python -m swotbot train
```

The `train` command runs five-fold cross-validation, prints accuracy and
macro-F1, fits the model on all 2,500 passages, and saves it to `models/`
along with metadata: the library version, a hash of the training data, and the
cross-validation scores. When the service starts, it checks this metadata and
retrains automatically if the model is missing, was built with a different
library version, or the training data has changed.

## Running the service

```bash
python -m swotbot serve          # http://127.0.0.1:8088, interactive docs at /docs
```

The service exposes three endpoints. `POST /predict` accepts a JSON body of the
form `{"texts": [...]}` with up to 256 passages and returns, for each one, the
label, the confidence and the probabilities of all four classes. `POST /api/`
keeps the original contract used by the Word add-in: it reads the passage from
the `TEXT` query parameter and returns a pair such as `["THREATS", 52.22]`,
where the number is the confidence in percent. `GET /health` reports the
service status, the class list and the model's cross-validation scores.

```python
import requests

r = requests.post(
    "http://127.0.0.1:8088/predict",
    json={"texts": ["Manpower attrition risk along with increasing salary cost."]},
)
print(r.json())
```

## Limitations

The classifier makes mistakes. The most common confusions are between strengths
and weaknesses, which share vocabulary but point in opposite directions, and
between weaknesses and threats, which differ mainly in whether the risk is
internal or external. Its output should be treated as a first draft for an
analyst to review, not as a finished report.

The training data comes from a single genre: Indian credit-rating narratives.
Accuracy on press releases, blogs or legal text will probably be lower until
labelled examples from those sources are added.

The confidence score is a model probability, not a guarantee. Passages with low
confidence, for example below 0.5, are best routed to a person.

## What would improve it most

The largest gains are likely to come from data rather than from model changes.
Relabelling `test.csv` and adding passages from the other document types
would broaden coverage. A held-out test set labelled independently and never
used for model selection would make the reported accuracy trustworthy. Finally,
generating the Word "SWOT Analysis" document on the server, for example with
`python-docx`, would let the pipeline run end to end without the VBA add-in.
