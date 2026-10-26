"""Command line: `python -m swotbot {train,predict,serve}`."""

import argparse
import json
import logging
from pathlib import Path

from .config import settings
from .model import SwotClassifier


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(prog="swotbot")
    sub = parser.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="train, cross-validate and save the model")
    t.add_argument("--data", type=Path, default=settings.data_path)
    t.add_argument("--out", type=Path, default=settings.model_dir)
    t.add_argument("--cv", type=int, default=5, help="CV folds (0 to skip)")

    p = sub.add_parser("predict", help="classify texts given as arguments")
    p.add_argument("texts", nargs="+")

    s = sub.add_parser("serve", help="run the HTTP API")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8088)
    s.add_argument("--workers", type=int, default=1)

    args = parser.parse_args()
    if args.cmd == "train":
        model = SwotClassifier.train(args.data, cv_folds=args.cv)
        model.save(args.out)
        print(json.dumps(model.meta, indent=2))
    elif args.cmd == "predict":
        model = SwotClassifier.load_or_train(settings.model_dir, settings.data_path)
        for text, pred in zip(args.texts, model.predict(args.texts)):
            print(f"{pred.label:13s} {pred.confidence:6.1%}  {text[:80]}")
    else:
        import uvicorn

        uvicorn.run("swotbot.api:app", host=args.host, port=args.port, workers=args.workers)


if __name__ == "__main__":
    main()
