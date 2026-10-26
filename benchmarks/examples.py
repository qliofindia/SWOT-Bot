"""Print out-of-fold predictions of the production model on real passages.

Each passage is classified by a model trained on the other four folds, so the
model has never seen it. Shows correct predictions, errors and a low-confidence
case, chosen with fixed seeds so the output is reproducible.

    python benchmarks/examples.py
"""

import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from swotbot.model import LABELS, build_pipeline, load_dataset  # noqa: E402

X, y = load_dataset(Path(__file__).resolve().parent.parent / "data" / "train.csv")
X, y = X.reset_index(drop=True), y.reset_index(drop=True)
cv = StratifiedKFold(5, shuffle=True, random_state=42)
proba = cross_val_predict(build_pipeline(), X, y, cv=cv, method="predict_proba")
classes = np.array(sorted(y.unique()))
df = pd.DataFrame({"text": X, "true": y, "pred": classes[proba.argmax(1)], "conf": proba.max(1)})
print(f"Out-of-fold accuracy on {len(df)} passages: {(df.true == df.pred).mean():.3f}\n")

short = df[df.text.str.len().between(61, 229)]
picks = [short[(short.true == c) & (short.pred == c) & short.conf.between(0.6, 0.9)].sample(1, random_state=7) for c in LABELS]
picks.append(short[short.true != short.pred].sample(3, random_state=3))
picks.append(short[(short.true == short.pred) & (short.conf < 0.45)].sample(1, random_state=1))

for i, (idx, r) in enumerate(pd.concat(picks).iterrows(), 1):
    mark = "correct" if r.true == r.pred else "WRONG"
    print(f"[{i}] row {idx}  predicted {r.pred} ({r.conf:.2f})  labelled {r.true}  -> {mark}")
    print(textwrap.indent(textwrap.fill(r.text, 96), "    "), "\n")
