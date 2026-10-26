# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "torch==2.14.1",
#     "transformers==5.20.2",
#     "sentence-transformers==6.2.0",
#     "accelerate",
#     "scikit-learn==1.9.1",
#     "pandas==3.1.0",
#     "numpy",
#     "scipy",
#     "joblib",
#     "pydantic-settings",
#     "lightgbm==4.7.0",
#     "gensim==4.4.0",
#     "fasttext==0.9.3",
#     "spacy==3.8.16",
#     "en_core_web_md @ https://github.com/explosion/spacy-models/releases/download/en_core_web_md-3.8.0/en_core_web_md-3.8.0-py3-none-any.whl",
#     "nbformat==5.11.1",
#     "nbclient==0.11.0",
#     "ipykernel==7.4.0",
# ]
# ///
"""Execute notebooks/benchmark.ipynb cell by cell as a Hugging Face Job.

Models with saved results in /bucket/results/folds are loaded, missing ones are
trained (RUN_MISSING=1). The executed notebook, with every cell's output, is
saved to /bucket/results/benchmark.executed.ipynb after each cell, so a crash
still leaves the output of every cell that ran.
"""

import os
import shutil
import subprocess
from pathlib import Path

import nbformat
from nbclient import NotebookClient

BUCKET = Path(os.environ.get("BUCKET_MOUNT", "/bucket"))
WORK = Path("/work")
OUT = BUCKET / "results" / "benchmark.executed.ipynb"

subprocess.run(["nvidia-smi", "-L"], check=False)
shutil.copytree(BUCKET / "code", WORK, dirs_exist_ok=True)
os.environ.update(SWOTBOT_BENCH_OUT=str(BUCKET / "results"), RUN_MISSING="1", PYTHONPATH=str(WORK))

nb = nbformat.read(WORK / "notebooks" / "benchmark.ipynb", as_version=4)
client = NotebookClient(nb, timeout=None, kernel_name="python3", allow_errors=True,
                        resources={"metadata": {"path": str(WORK / "notebooks")}})
with client.setup_kernel():
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        print(f"--- cell {i}: {cell.source.splitlines()[0][:90]}", flush=True)
        client.execute_cell(cell, i)
        for o in cell.get("outputs", []):
            text = o.get("text") or o.get("data", {}).get("text/plain", "")
            if o.get("output_type") == "error":
                text = f"{o['ename']}: {o['evalue']}"
            print(str(text)[:600], flush=True)
        nbformat.write(nb, OUT)
print(f"saved {OUT}")
