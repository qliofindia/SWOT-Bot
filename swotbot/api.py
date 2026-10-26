"""HTTP API for the SWOT classifier.

`/api/` keeps the contract the Word add-in (word-addin/httpRequest.bas) uses:
POST /api/?TEXT=<text>  ->  ["<LABEL>", <confidence in percent>]
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Annotated
from urllib.parse import unquote

from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, Field

from . import __version__
from .config import settings
from .model import SwotClassifier

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.classifier = SwotClassifier.load_or_train(settings.model_dir, settings.data_path)
    log.info("model ready: %s", app.state.classifier.meta.get("cv"))
    yield


app = FastAPI(title="SWOT-Bot", version=__version__, lifespan=lifespan)


class PredictRequest(BaseModel):
    texts: list[Annotated[str, Field(min_length=1, max_length=settings.max_text_chars)]] = Field(
        min_length=1, max_length=settings.max_batch
    )


class PredictionOut(BaseModel):
    text: str
    label: str
    confidence: float
    scores: dict[str, float]


def _classifier(request: Request) -> SwotClassifier:
    return request.app.state.classifier


def _clean_legacy_text(raw: str) -> str:
    # The VBA client sends Python-list-looking strings such as "['some text']".
    text = unquote(raw).strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1].strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "'\"":
        text = text[1:-1]
    return text.strip()


@app.get("/health")
def health(request: Request) -> dict:
    clf = _classifier(request)
    return {"status": "ok", "version": __version__, "classes": clf.classes, "model": clf.meta}


@app.post("/predict", response_model=list[PredictionOut])
def predict(body: PredictRequest, request: Request) -> list[PredictionOut]:
    preds = _classifier(request).predict(body.texts)
    return [
        PredictionOut(text=t, label=p.label, confidence=round(p.confidence, 4), scores=p.scores)
        for t, p in zip(body.texts, preds)
    ]


@app.post("/api/")
def legacy_predict(
    request: Request,
    TEXT: Annotated[str | None, Query(max_length=settings.max_text_chars * 3)] = None,
) -> list:
    if not TEXT:
        raise HTTPException(400, "No TEXT field provided.")
    text = _clean_legacy_text(TEXT)
    if not text:
        raise HTTPException(400, "TEXT is empty.")
    p = _classifier(request).predict([text])[0]
    return [p.label, round(p.confidence * 100, 2)]
