"""Generate notebooks/benchmark.ipynb from the model registry (one cell per model).

    python notebooks/make_benchmark_notebook.py
"""

import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from benchmarks.benchmark import registry  # noqa: E402

nb = nbf.v4.new_notebook()
cells = nb.cells


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text):
    cells.append(nbf.v4.new_code_cell(text))


md("""# SWOT-Bot model benchmark, step by step

One cell per model, run top to bottom. Each cell:

1. loads the model's saved per-fold results from `RESULTS_DIR/folds/` if they exist (nothing is retrained);
2. otherwise, when `RUN_MISSING = True`, trains and evaluates the model on the shared 5 folds, saves the scores, and shows them;
3. on an error, prints the traceback in that cell and carries on, so the problem can be fixed and that cell rerun alone.

Results are saved per model, so a kernel crash (out of memory, GPU fault) loses only the model that was running.
The same functions back `benchmarks/benchmark.py`.""")

code("""import os, sys, time, traceback
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
os.environ.setdefault("SWOTBOT_BENCH_OUT", str(ROOT / "benchmarks"))

# Train models that have no saved results. Keep False on CPU: BERT-size models take hours there.
RUN_MISSING = os.environ.get("RUN_MISSING", "0") == "1"

import pandas as pd
from IPython.display import display
from sklearn.model_selection import StratifiedKFold
from benchmarks import benchmark as B

X, y = B.load_dataset(ROOT / "data" / "train.csv")
X, y = X.reset_index(drop=True), y.reset_index(drop=True)
SPLITS = list(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y))
REG = B.registry()
print(f"{len(X)} passages, {len(REG)} models, results in {B.FOLDS_DIR}, RUN_MISSING={RUN_MISSING}")
print(y.value_counts().to_string())""")

code('''def step(name):
    """Load or run one model and show its per-fold scores. Errors are printed, not raised."""
    suite, family, factory = REG[name]
    path = B.FOLDS_DIR / f"{B.slug(name)}.csv"
    if path.exists():
        df, how = pd.read_csv(path), "loaded saved results"
    elif not RUN_MISSING:
        print(f"{name}: no saved results (set RUN_MISSING=1 to train it)")
        return None
    else:
        t0 = time.perf_counter()
        try:
            df = B.run_model(name, factory, X, y, SPLITS, family)
        except Exception:
            print(f"{name}: FAILED")
            traceback.print_exc()
            return None
        how = f"trained in {time.perf_counter() - t0:.0f}s"
    print(f"{name}  [{family}]  {how}")
    print(f"macro-F1 {df.f1_macro.mean():.4f} ± {df.f1_macro.std():.4f}   accuracy {df.accuracy.mean():.4f}   "
          f"log loss {df.log_loss.mean():.3f}   fit {df.fit_s.mean():.1f} s/fold   predict {df.ms_per_doc.iloc[0]:.2f} ms/doc")
    display(df[["fold", "f1_macro", "accuracy", "log_loss", "fit_s"]].round(4))
    return df''')

family = None
for name, (_, fam, _) in registry().items():
    if fam != family:
        md(f"## {fam}")
        family = fam
    code(f"_ = step({name!r})")

md("""## Ranking

Models with all 5 folds, sorted by macro-F1. `p_vs_best` is the Nadeau-Bengio corrected resampled
t-test against the top model; p ≥ 0.05 means the gap is within fold-to-fold noise.""")
code("""ranking = B.report(len(X), 5)
cols = ["model", "family", "f1_macro", "f1_std", "accuracy", "log_loss", "p_vs_best", "fit_s", "ms_per_doc", "size_kb"]
display(ranking[cols].round(4))""")

nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nb.metadata["language_info"] = {"name": "python"}
out = ROOT / "notebooks" / "benchmark.ipynb"
nbf.write(nb, out)
print(f"wrote {out.relative_to(ROOT)} ({len(cells)} cells)")
