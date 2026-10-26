"""Benchmark SWOT classifiers on data/train.csv: linear, embedding, neural and transformer models.

Protocol: stratified K-fold (default 5 folds, seed 42); every model sees exactly
the same folds. Per-fold scores are written to benchmarks/folds/<model>.csv, so
slow models (BERT) can run separately or resume; `report` merges whatever exists.

    python benchmarks/benchmark.py list
    python benchmarks/benchmark.py run --suite fast          # linear + embeddings + neural
    python benchmarks/benchmark.py run --suite transformers  # BERT-family fine-tuning (slow on CPU)
    python benchmarks/benchmark.py run --models "fastText (scratch)" "CNN (Kim) + GloVe"
    python benchmarks/benchmark.py report                    # -> RESULTS.md, results.csv

Models are compared to the top model with the Nadeau-Bengio corrected
resampled t-test (a plain paired t-test over CV folds is overconfident because
training sets overlap).
"""

from __future__ import annotations

import argparse
import io
import math
import os
import re
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline, make_pipeline, make_union
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("SWOTBOT_BENCH_OUT", ROOT / "benchmarks"))
FOLDS_DIR = OUT / "folds"
sys.path.insert(0, str(ROOT))
from swotbot.estimators import NBSVM, SpacyVectors  # noqa: E402
from swotbot.model import build_pipeline, load_dataset  # noqa: E402

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------- model registry
def _word():
    return TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, strip_accents="unicode")


def _word_char():
    return make_union(_word(), TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2))


def _cal_svc(C):
    return CalibratedClassifierCV(LinearSVC(C=C), cv=5)


def _lgbm():
    from lightgbm import LGBMClassifier

    return LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, verbose=-1, random_state=0)


def _nn():
    from benchmarks import neural  # torch is imported only when neural models run

    return neural


def registry() -> dict[str, tuple[str, str, callable]]:
    """name -> (suite, family, factory)."""
    R = {}

    def add(suite, family, name, factory):
        R[name] = (suite, family, factory)

    # Sparse linear / classical
    add("fast", "Linear (TF-IDF)", "TF-IDF + LogReg (current production)", build_pipeline)
    add("fast", "Linear (TF-IDF)", "TF-IDF word+char + LogReg", lambda: make_pipeline(_word_char(), LogisticRegression(C=10, max_iter=3000)))
    add("fast", "Linear (TF-IDF)", "TF-IDF + LinearSVC (calibrated)", lambda: make_pipeline(_word(), _cal_svc(1.0)))
    add("fast", "Linear (TF-IDF)", "TF-IDF word+char + LinearSVC (calibrated)", lambda: make_pipeline(_word_char(), _cal_svc(0.5)))
    add("fast", "Linear (TF-IDF)", "TF-IDF + SGD (modified Huber)", lambda: make_pipeline(_word(), SGDClassifier(loss="modified_huber", alpha=1e-4, max_iter=50, random_state=0)))
    add("fast", "Linear (TF-IDF)", "TF-IDF + Ridge", lambda: make_pipeline(_word(), RidgeClassifier(alpha=1.0)))
    add("fast", "Probabilistic", "TF-IDF + Complement Naive Bayes", lambda: make_pipeline(_word(), ComplementNB(alpha=0.3)))
    add("fast", "Probabilistic", "NB-SVM (NB-weighted LogReg, 1-3 grams)", lambda: make_pipeline(CountVectorizer(ngram_range=(1, 3), binary=True, min_df=2, strip_accents="unicode"), NBSVM(C=4.0)))
    add("fast", "Tree ensemble", "LSA-300 + LightGBM", lambda: make_pipeline(_word(), TruncatedSVD(300, random_state=0), _lgbm()))
    add("fast", "Linear (TF-IDF)", "LSA-300 + LogReg", lambda: make_pipeline(_word(), TruncatedSVD(300, random_state=0), StandardScaler(), LogisticRegression(C=0.1, max_iter=3000)))
    add("fast", "Ensemble", "Soft vote: word+char LogReg + cal. SVC + NB-SVM", lambda: VotingClassifier(
        [
            ("lr", make_pipeline(_word_char(), LogisticRegression(C=10, max_iter=3000))),
            ("svc", make_pipeline(_word_char(), _cal_svc(0.5))),
            ("nbsvm", make_pipeline(CountVectorizer(ngram_range=(1, 3), binary=True, min_df=2), NBSVM())),
        ],
        voting="soft",
    ))

    # Static word embeddings
    add("fast", "Word embeddings", "spaCy en_core_web_md vectors + LogReg", lambda: make_pipeline(SpacyVectors(), StandardScaler(), LogisticRegression(C=0.1, max_iter=3000)))
    add("fast", "Word embeddings", "Gensim GloVe-300 mean + LogReg", lambda: make_pipeline(_nn().GloveMean(), StandardScaler(), LogisticRegression(C=0.1, max_iter=3000)))
    add("fast", "Word embeddings", "Gensim Doc2Vec (PV-DBOW) + LogReg", lambda: make_pipeline(_nn().Doc2VecVectorizer(), StandardScaler(), LogisticRegression(C=0.1, max_iter=3000)))
    add("fast", "Word embeddings", "TF-IDF + GloVe-300 mean + LogReg", lambda: make_pipeline(
        make_union(_word(), make_pipeline(_nn().GloveMean(), StandardScaler(with_mean=False))), LogisticRegression(C=3, max_iter=3000)))

    # fastText
    add("fast", "fastText", "fastText (scratch, bigrams)", lambda: _nn().FastTextClassifier())
    add("fast", "fastText", "fastText (GloVe-300 init, bigrams)", lambda: _nn().FastTextClassifier(pretrained="glove-wiki-gigaword-300", lr=0.3))

    # Neural networks trained on this data
    add("fast", "Neural (from scratch)", "ANN / MLP on TF-IDF", lambda: _nn().TfidfMLPClassifier())
    add("fast", "Neural (from scratch)", "CNN (Kim 2014) + GloVe", lambda: _nn().TorchTextClassifier(arch="cnn"))
    add("fast", "Neural (from scratch)", "BiLSTM + attention + GloVe", lambda: _nn().TorchTextClassifier(arch="bilstm"))
    add("fast", "Neural (from scratch)", "Transformer encoder (2-layer, from scratch) + GloVe", lambda: _nn().TorchTextClassifier(arch="transformer"))

    # Pretrained transformers, frozen (embeddings + LogReg)
    for model, label in [
        ("sentence-transformers/all-MiniLM-L6-v2", "MiniLM-L6"),
        ("sentence-transformers/all-mpnet-base-v2", "MPNet-base"),
        ("BAAI/bge-base-en-v1.5", "BGE-base"),
    ]:
        add("fast", "Transformer (frozen)", f"{label} sentence embeddings + LogReg",
            lambda m=model: make_pipeline(_nn().SentenceEncoder(m), LogisticRegression(C=10, max_iter=3000)))

    # Pretrained transformers, fine-tuned end to end
    add("transformers", "Transformer (fine-tuned)", "DistilBERT-base fine-tuned", lambda: _nn().HFFineTuneClassifier("distilbert-base-uncased", lr=5e-5))
    add("transformers", "Transformer (fine-tuned)", "BERT-base fine-tuned", lambda: _nn().HFFineTuneClassifier("bert-base-uncased"))
    add("bert-extra", "Transformer (fine-tuned)", "FinBERT (ProsusAI) fine-tuned", lambda: _nn().HFFineTuneClassifier("ProsusAI/finbert"))
    add("bert-extra", "Transformer (fine-tuned)", "RoBERTa-base fine-tuned", lambda: _nn().HFFineTuneClassifier("roberta-base", lr=2e-5))
    return R


# ---------------------------------------------------------------- evaluation
def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def size_kb(est) -> float:
    """Size of the fitted artifact; pretrained resources count via their parameter size."""
    if hasattr(est, "size_kb"):
        return est.size_kb()
    if isinstance(est, Pipeline):
        return sum(size_kb(step) for _, step in est.steps)
    if hasattr(est, "transformer_list"):
        return sum(size_kb(t) for _, t in est.transformer_list)
    if isinstance(est, SpacyVectors):
        import spacy  # noqa: F401  # vectors table of en_core_web_md

        return 20000 * 300 * 4 / 1024
    buf = io.BytesIO()
    joblib.dump(est, buf, compress=3)
    return buf.tell() / 1024


def run_model(name, factory, X, y, splits, family):
    rows, est = [], None
    for k, (tr, te) in enumerate(splits):
        est = factory()
        t0 = time.perf_counter()
        est.fit(X.iloc[tr], y.iloc[tr])
        fit_s = time.perf_counter() - t0
        pred = est.predict(X.iloc[te])
        ll = log_loss(y.iloc[te], est.predict_proba(X.iloc[te]), labels=est.classes_) if hasattr(est, "predict_proba") else math.nan
        rows.append({"fold": k, "accuracy": accuracy_score(y.iloc[te], pred), "f1_macro": f1_score(y.iloc[te], pred, average="macro"),
                     "log_loss": ll, "fit_s": fit_s})
        print(f"    fold {k}: f1={rows[-1]['f1_macro']:.4f}  fit {fit_s:.0f}s", flush=True)
    # Latency on unseen strings (defeats embedding caches), single process.
    sample = [t + " ." for t in X.iloc[splits[-1][1][:200]]]
    t0 = time.perf_counter()
    est.predict(sample)
    ms_per_doc = (time.perf_counter() - t0) / len(sample) * 1000
    df = pd.DataFrame(rows).assign(model=name, family=family, ms_per_doc=ms_per_doc, size_kb=size_kb(est))
    FOLDS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(FOLDS_DIR / f"{slug(name)}.csv", index=False)
    return df


def corrected_ttest(a, b, n_train, n_test) -> float:
    """Nadeau & Bengio (2003) corrected resampled t-test; two-sided p-value."""
    d = np.asarray(a) - np.asarray(b)
    if len(d) < 2 or d.var(ddof=1) == 0:
        return 1.0
    t = d.mean() / math.sqrt((1 / len(d) + n_test / n_train) * d.var(ddof=1))
    return float(2 * stats.t.sf(abs(t), df=len(d) - 1))


def report(n_samples: int, folds: int) -> pd.DataFrame:
    files = sorted(FOLDS_DIR.glob("*.csv"))
    if not files:
        sys.exit("no fold results yet; run `benchmark.py run` first")
    allf = pd.concat([pd.read_csv(f) for f in files])
    allf = allf[allf.groupby("model").fold.transform("count") == folds]  # complete runs only
    g = allf.groupby(["model", "family"])
    df = g.agg(f1_macro=("f1_macro", "mean"), f1_std=("f1_macro", "std"), accuracy=("accuracy", "mean"),
               accuracy_std=("accuracy", "std"), log_loss=("log_loss", "mean"), fit_s=("fit_s", "mean"),
               ms_per_doc=("ms_per_doc", "first"), size_kb=("size_kb", "first")).reset_index()
    df = df.sort_values("f1_macro", ascending=False).reset_index(drop=True)
    best = df.model.iloc[0]
    per = {m: f.sort_values("fold").f1_macro.values for m, f in allf.groupby("model")}
    n_test = n_samples // folds
    df["p_vs_best"] = [math.nan if m == best else corrected_ttest(per[best], per[m], n_samples - n_test, n_test) for m in df.model]
    df.to_csv(OUT / "results.csv", index=False, float_format="%.5f")
    (OUT / "RESULTS.md").write_text(render_md(df, n_samples, folds))
    return df


def _fmt_size(kb: float) -> str:
    return f"{kb / 1024:.0f} MB" if kb >= 1024 else f"{kb:.0f} KB"


def render_md(df: pd.DataFrame, n: int, folds: int) -> str:
    lines = [
        "# Model benchmark",
        "",
        f"Data: `data/train.csv`, {n} passages, 4 balanced-ish classes "
        "(S 827 / W 619 / O 549 / T 505). "
        f"Protocol: {folds}-fold stratified CV, seed 42, identical folds for every model. "
        "Generated by `python benchmarks/benchmark.py report`.",
        "",
        "| # | Model | Family | Macro-F1 | Accuracy | Log loss | p vs #1 | Fit / fold | Predict | Size |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for i, r in df.iterrows():
        ll = "n/a" if math.isnan(r.log_loss) else f"{r.log_loss:.3f}"
        p = "—" if math.isnan(r.p_vs_best) else f"{r.p_vs_best:.3f}"
        fit = f"{r.fit_s:.1f} s" if r.fit_s < 120 else f"{r.fit_s / 60:.0f} min"
        lines.append(f"| {i + 1} | {r.model} | {r.family} | {r.f1_macro:.3f} ± {r.f1_std:.3f} | {r.accuracy:.3f} ± {r.accuracy_std:.3f} "
                     f"| {ll} | {p} | {fit} | {r.ms_per_doc:.2f} ms/doc | {_fmt_size(r.size_kb)} |")
    lines += [
        "",
        "Notes",
        "",
        "- ± is the standard deviation across folds.",
        "- *p vs #1*: corrected resampled t-test on per-fold macro-F1 against the top row. "
        "p ≥ 0.05 means the gap is not distinguishable from fold noise with this much data.",
        "- Log loss measures probability quality (lower is better); n/a for models without probabilities.",
        "- Fit and predict times are measured on the machine that ran the benchmark (CPU unless stated). "
        "Predict is per document on unseen strings, batch of 200.",
        "- Size is the trained artifact; for pretrained encoders/vectors it is their parameter memory.",
        "",
    ]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--suite", choices=["fast", "transformers", "bert-extra", "all"])
    r.add_argument("--models", nargs="+")
    r.add_argument("--force", action="store_true", help="re-run models that already have results")
    sub.add_parser("report")
    sub.add_parser("list")
    for p in sub.choices.values():
        p.add_argument("--data", type=Path, default=ROOT / "data" / "train.csv")
        p.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    reg = registry()
    if args.cmd == "list":
        for name, (suite, family, _) in reg.items():
            done = "done" if (FOLDS_DIR / f"{slug(name)}.csv").exists() else ""
            print(f"{suite:13s} {family:26s} {name:55s} {done}")
        return

    X, y = load_dataset(args.data)
    X, y = X.reset_index(drop=True), y.reset_index(drop=True)
    if args.cmd == "run":
        if args.models:
            unknown = set(args.models) - set(reg)
            if unknown:
                sys.exit(f"unknown models: {sorted(unknown)}")
            names = args.models
        else:
            suites = {"fast", "transformers", "bert-extra"} if args.suite == "all" else {args.suite or "fast"}
            names = [n for n, (s, _, _) in reg.items() if s in suites]
        splits = list(StratifiedKFold(args.folds, shuffle=True, random_state=42).split(X, y))
        for name in names:
            if not args.force and (FOLDS_DIR / f"{slug(name)}.csv").exists():
                print(f"skip (done): {name}")
                continue
            suite, family, factory = reg[name]
            print(f"== {name}", flush=True)
            t0 = time.perf_counter()
            try:
                df = run_model(name, factory, X, y, splits, family)
            except Exception as exc:  # keep going; one broken model must not kill the run
                print(f"   FAILED: {type(exc).__name__}: {exc}", flush=True)
                continue
            print(f"   macro-F1 {df.f1_macro.mean():.4f}  acc {df.accuracy.mean():.4f}  ({time.perf_counter() - t0:.0f}s)", flush=True)
    df = report(len(X), args.folds)
    print(df[["model", "f1_macro", "accuracy", "log_loss", "p_vs_best", "fit_s", "ms_per_doc"]].to_string(float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
