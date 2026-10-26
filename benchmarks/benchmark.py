"""Benchmark candidate SWOT classifiers on data/train.csv.

Protocol: repeated stratified K-fold (default 5 folds x 3 repeats), identical
folds for every model. Reports accuracy, macro-F1, log loss, fit time,
inference latency and serialized model size, then compares every model with
the best one using the Nadeau-Bengio corrected resampled t-test (plain paired
t-tests over CV folds are overconfident because training sets overlap).

    python benchmarks/benchmark.py [--folds 5] [--repeats 3] [--no-spacy]

Writes benchmarks/results.csv and benchmarks/RESULTS.md.
"""

from __future__ import annotations

import argparse
import io
import math
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
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from swotbot.estimators import NBSVM, SpacyVectors  # noqa: E402
from swotbot.model import build_pipeline, load_dataset  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------- candidates
def word():
    return TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, strip_accents="unicode")


def word_char():
    return make_union(
        word(), TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2)
    )


def cal_svc(C=0.5):
    return CalibratedClassifierCV(LinearSVC(C=C), cv=5)


def candidates(use_spacy: bool) -> dict:
    m = {
        "tfidf-word + LogReg (production)": build_pipeline(),
        "tfidf-word+char + LogReg": make_pipeline(word_char(), LogisticRegression(C=10, max_iter=3000)),
        "tfidf-word + LinearSVC (calibrated)": make_pipeline(word(), cal_svc(1.0)),
        "tfidf-word+char + LinearSVC (calibrated)": make_pipeline(word_char(), cal_svc(0.5)),
        "tfidf-word + SGD (modified Huber)": make_pipeline(
            word(), SGDClassifier(loss="modified_huber", alpha=1e-4, max_iter=50, random_state=0)
        ),
        "tfidf-word + ComplementNB": make_pipeline(word(), ComplementNB(alpha=0.3)),
        "tfidf-word + Ridge": make_pipeline(word(), RidgeClassifier(alpha=1.0)),
        "NB-SVM (binary 1-3 grams)": make_pipeline(
            CountVectorizer(ngram_range=(1, 3), binary=True, min_df=2, strip_accents="unicode"),
            NBSVM(C=4.0),
        ),
        "LSA-300 + LogReg": make_pipeline(
            word(), TruncatedSVD(300, random_state=0), StandardScaler(), LogisticRegression(C=0.1, max_iter=3000)
        ),
        "LSA-300 + LightGBM": make_pipeline(word(), TruncatedSVD(300, random_state=0), _lgbm()),
        "Soft vote: word+char LogReg, cal. SVC, NB-SVM": VotingClassifier(
            [
                ("lr", make_pipeline(word_char(), LogisticRegression(C=10, max_iter=3000))),
                ("svc", make_pipeline(word_char(), cal_svc(0.5))),
                ("nbsvm", make_pipeline(CountVectorizer(ngram_range=(1, 3), binary=True, min_df=2), NBSVM())),
            ],
            voting="soft",
        ),
    }
    if use_spacy:
        m["spaCy vectors + LogReg"] = make_pipeline(
            SpacyVectors(), StandardScaler(), LogisticRegression(C=0.1, max_iter=3000)
        )
        m["tfidf-word + spaCy vectors + LogReg"] = make_pipeline(
            make_union(word(), make_pipeline(SpacyVectors(), StandardScaler(with_mean=False))),
            LogisticRegression(C=3, max_iter=3000),
        )
    return m


def _lgbm():
    from lightgbm import LGBMClassifier

    return LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, verbose=-1, random_state=0)


# ---------------------------------------------------------------- evaluation
def evaluate(model, X, y, splits):
    rows = []
    for train, test in splits:
        est = clone(model)
        t0 = time.perf_counter()
        est.fit(X.iloc[train], y.iloc[train])
        fit_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        pred = est.predict(X.iloc[test])
        pred_s = time.perf_counter() - t0
        ll = math.nan
        if hasattr(est, "predict_proba"):
            ll = log_loss(y.iloc[test], est.predict_proba(X.iloc[test]), labels=est.classes_)
        rows.append(
            {
                "accuracy": accuracy_score(y.iloc[test], pred),
                "f1_macro": f1_score(y.iloc[test], pred, average="macro"),
                "log_loss": ll,
                "fit_s": fit_s,
                "ms_per_1k": pred_s / len(test) * 1e6,
            }
        )
    return pd.DataFrame(rows), est


def model_size_kb(est) -> float:
    buf = io.BytesIO()
    joblib.dump(est, buf, compress=3)
    return buf.tell() / 1024


def corrected_ttest(a, b, n_train, n_test):
    """Nadeau & Bengio (2003) corrected resampled t-test; returns two-sided p-value."""
    d = np.asarray(a) - np.asarray(b)
    k = len(d)
    var = d.var(ddof=1)
    if var == 0:
        return 1.0
    t = d.mean() / math.sqrt((1 / k + n_test / n_train) * var)
    return float(2 * stats.t.sf(abs(t), df=k - 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "train.csv")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--no-spacy", action="store_true")
    ap.add_argument("--jobs", type=int, default=-1)
    args = ap.parse_args()

    X, y = load_dataset(args.data)
    X, y = X.reset_index(drop=True), y.reset_index(drop=True)
    cv = RepeatedStratifiedKFold(n_splits=args.folds, n_repeats=args.repeats, random_state=42)
    splits = list(cv.split(X, y))
    n_test = len(X) // args.folds
    n_train = len(X) - n_test

    models = candidates(not args.no_spacy)
    # spaCy pipelines share an in-process cache, so they run serially.
    parallel = {k: v for k, v in models.items() if "spaCy" not in k}
    serial = {k: v for k, v in models.items() if "spaCy" in k}

    def run(name, model):
        t0 = time.perf_counter()
        folds, last = evaluate(model, X, y, splits)
        print(f"  {name}: acc={folds.accuracy.mean():.4f}  ({time.perf_counter() - t0:.0f}s)", flush=True)
        return name, folds, model_size_kb(last)

    print(f"{len(X)} samples, {args.folds}x{args.repeats} CV, {len(models)} models")
    results = joblib.Parallel(n_jobs=args.jobs)(joblib.delayed(run)(n, m) for n, m in parallel.items())
    results += [run(n, m) for n, m in serial.items()]
    if serial:
        # Embedding cost dominates spaCy inference; measure it on uncached text.
        sample = [t + " ." for t in X.iloc[:1000]]
        t0 = time.perf_counter()
        SpacyVectors().transform(sample)
        spacy_ms = (time.perf_counter() - t0) / len(sample) * 1e6

    folds_by = {n: f for n, f, _ in results}
    summary = []
    for name, f, size in results:
        ms = f.ms_per_1k.mean() + (spacy_ms if "spaCy" in name else 0)
        summary.append(
            {
                "model": name,
                "accuracy": f.accuracy.mean(),
                "accuracy_std": f.accuracy.std(),
                "f1_macro": f.f1_macro.mean(),
                "f1_std": f.f1_macro.std(),
                "log_loss": f.log_loss.mean(),
                "fit_s": f.fit_s.mean(),
                "ms_per_1k_docs": ms,
                "size_kb": size,
            }
        )
    df = pd.DataFrame(summary).sort_values("f1_macro", ascending=False).reset_index(drop=True)
    best = df.model.iloc[0]
    df["p_vs_best"] = [
        math.nan if m == best else corrected_ttest(folds_by[best].f1_macro, folds_by[m].f1_macro, n_train, n_test)
        for m in df.model
    ]

    out = ROOT / "benchmarks"
    df.to_csv(out / "results.csv", index=False, float_format="%.5f")
    (out / "RESULTS.md").write_text(render_md(df, len(X), args))
    print(df.to_string(float_format=lambda v: f"{v:.4f}"))


def render_md(df: pd.DataFrame, n: int, args) -> str:
    lines = [
        "# Model benchmark",
        "",
        f"Data: `data/train.csv`, {n} labelled passages, 4 classes. "
        f"Protocol: {args.folds}-fold stratified CV x {args.repeats} repeats (same folds for every model). "
        "Generated by `python benchmarks/benchmark.py`.",
        "",
        "| # | Model | Macro-F1 | Accuracy | Log loss | p vs best | Fit (s) | Predict (ms / 1k docs) | Size (KB) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for i, r in df.iterrows():
        ll = "n/a" if math.isnan(r.log_loss) else f"{r.log_loss:.3f}"
        p = "best" if math.isnan(r.p_vs_best) else f"{r.p_vs_best:.3f}"
        lines.append(
            f"| {i + 1} | {r.model} | {r.f1_macro:.3f} ± {r.f1_std:.3f} | {r.accuracy:.3f} ± {r.accuracy_std:.3f} "
            f"| {ll} | {p} | {r.fit_s:.2f} | {r.ms_per_1k_docs:.0f} | {r.size_kb:.0f} |"
        )
    lines += [
        "",
        "± is the standard deviation across folds. *p vs best*: Nadeau-Bengio corrected resampled t-test "
        "on per-fold macro-F1; p ≥ 0.05 means the difference from the top model is not statistically "
        "distinguishable on this data. Log loss is n/a for models without probabilities (Ridge). "
        "Predict time is measured on held-out folds in one process and includes spaCy parsing where used.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
