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
# ]
# ///
"""Entry point for running the benchmark as a Hugging Face Job.

The bucket is mounted at /bucket: code is read from /bucket/code, and per-fold
results are written to /bucket/results as each model finishes, so a timeout
loses at most the model in progress.

    python benchmarks/launch_hf_job.py      # uploads code and starts this script
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

BUCKET = Path(os.environ.get("BUCKET_MOUNT", "/bucket"))
WORK = Path("/work")

subprocess.run(["nvidia-smi"], check=False)
shutil.copytree(BUCKET / "code", WORK, dirs_exist_ok=True)
env = {**os.environ, "SWOTBOT_BENCH_OUT": str(BUCKET / "results"), "PYTHONPATH": str(WORK)}
suite = sys.argv[1:] or ["--suite", "all"]
cmd = [sys.executable, "benchmarks/benchmark.py", "run", *suite]
print("+", " ".join(cmd), flush=True)
sys.exit(subprocess.run(cmd, cwd=WORK, env=env).returncode)
