"""Upload the benchmark code to a Hugging Face bucket and run it as an HF Job.

Requires HF_TOKEN (in the environment or .env) with bucket write and Jobs permissions.
Billed to the token's account at the flavor's rate.

    python benchmarks/launch_hf_job.py [--flavor a10g-small] [--timeout 2h] [-- --suite fast]
    python benchmarks/launch_hf_job.py --fetch     # download results into benchmarks/
"""

import argparse
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, Volume

ROOT = Path(__file__).resolve().parent.parent
CODE_FILES = ["pyproject.toml", "data/train.csv", "swotbot", "benchmarks/__init__.py",
              "benchmarks/benchmark.py", "benchmarks/neural.py", "benchmarks/hf_job.py"]


def load_dotenv():
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", default="sln2737/Brain")
    ap.add_argument("--flavor", default="a10g-small")
    ap.add_argument("--timeout", default="2h")
    ap.add_argument("--fetch", action="store_true", help="download results instead of launching")
    ap.add_argument("suite", nargs="*", help="arguments passed to `benchmark.py run`")
    args = ap.parse_args()
    load_dotenv()
    api = HfApi(token=os.environ["HF_TOKEN"])
    remote = f"hf://buckets/{args.bucket}"

    if args.fetch:
        api.sync_bucket(f"{remote}/results", str(ROOT / "benchmarks"), quiet=True)
        print("results synced into benchmarks/")
        return

    with tempfile.TemporaryDirectory() as tmp:
        for rel in CODE_FILES:  # explicit allow-list: never ships .env or other secrets
            src, dst = ROOT / rel, Path(tmp) / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.is_dir():
                shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__"))
            else:
                shutil.copy2(src, dst)
        api.sync_bucket(tmp, f"{remote}/code", delete=True, quiet=True)
    print(f"code uploaded to {remote}/code")

    job = api.run_uv_job(
        str(ROOT / "benchmarks" / "hf_job.py"),
        script_args=args.suite,
        flavor=args.flavor,
        timeout=args.timeout,
        secrets={"HF_TOKEN": os.environ["HF_TOKEN"]},
        volumes=[Volume(type="bucket", source=args.bucket, mount_path="/bucket")],
        name="swotbot-benchmark",
    )
    print(f"job {job.id} started on {args.flavor}: {getattr(job, 'url', '')}")


if __name__ == "__main__":
    main()
