"""Neural and embedding-based SWOT classifiers with a scikit-learn interface.

Every class implements fit / predict / predict_proba (classifiers) or
fit / transform (feature extractors), so they run inside the same CV harness
as the linear models. Pretrained resources (GloVe, sentence encoders, BERT)
are downloaded on first use and cached by gensim / Hugging Face.
"""

from __future__ import annotations

import math
import os
import random
import re
import tempfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

torch.set_num_threads(int(os.environ.get("SWOTBOT_THREADS", os.cpu_count() or 4)))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _params_kb(module: nn.Module) -> float:
    return sum(p.numel() * p.element_size() for p in module.parameters()) / 1024


# ------------------------------------------------------------------ GloVe
_GLOVE: dict = {}


def glove(name: str = "glove-wiki-gigaword-300"):
    """Pretrained GloVe KeyedVectors via gensim-data (downloaded once, ~380 MB)."""
    if name not in _GLOVE:
        import gensim.downloader

        _GLOVE[name] = gensim.downloader.load(name)
    return _GLOVE[name]


class GloveMean(TransformerMixin, BaseEstimator):
    """Average of pretrained GloVe vectors over in-vocabulary tokens (gensim)."""

    def __init__(self, name: str = "glove-wiki-gigaword-300"):
        self.name = name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        kv = glove(self.name)
        out = np.zeros((len(X), kv.vector_size), dtype=np.float32)
        for i, text in enumerate(X):
            toks = [t for t in tokenize(text) if t in kv.key_to_index]
            if toks:
                out[i] = kv[toks].mean(axis=0)
        return out

    def size_kb(self) -> float:
        kv = glove(self.name)
        return kv.vectors.nbytes / 1024


class Doc2VecVectorizer(TransformerMixin, BaseEstimator):
    """gensim Doc2Vec (PV-DBOW with word training) fitted on the training fold only."""

    def __init__(self, vector_size: int = 150, epochs: int = 40, seed: int = 0):
        self.vector_size = vector_size
        self.epochs = epochs
        self.seed = seed

    def fit(self, X, y=None):
        from gensim.models.doc2vec import Doc2Vec, TaggedDocument

        docs = [TaggedDocument(tokenize(t), [i]) for i, t in enumerate(X)]
        self.model_ = Doc2Vec(
            docs,
            vector_size=self.vector_size,
            dm=0,
            dbow_words=1,
            window=8,
            min_count=2,
            epochs=self.epochs,
            workers=1,  # determinism
            seed=self.seed,
        )
        return self

    def transform(self, X):
        return np.vstack([self.model_.infer_vector(tokenize(t), epochs=self.epochs) for t in X])

    def size_kb(self) -> float:
        return (self.model_.wv.vectors.nbytes + self.model_.dv.vectors.nbytes) / 1024


# ------------------------------------------------------------------ fastText
class FastTextClassifier(ClassifierMixin, BaseEstimator):
    """Facebook fastText supervised classifier, optionally initialised from GloVe."""

    def __init__(self, dim=100, epoch=30, lr=0.5, word_ngrams=2, pretrained=None, seed=0):
        self.dim = dim
        self.epoch = epoch
        self.lr = lr
        self.word_ngrams = word_ngrams
        self.pretrained = pretrained
        self.seed = seed

    @staticmethod
    def _clean(text: str) -> str:
        return " ".join(tokenize(text))

    def fit(self, X, y):
        import fasttext

        y = np.asarray(y)
        self.classes_ = np.unique(y)
        kwargs = {}
        with tempfile.TemporaryDirectory() as tmp:
            train = os.path.join(tmp, "train.txt")
            with open(train, "w") as f:
                for t, lab in zip(X, y):
                    f.write(f"__label__{lab} {self._clean(t)}\n")
            if self.pretrained:
                vec = os.path.join(tmp, "vectors.vec")
                kv = glove(self.pretrained)
                kv.save_word2vec_format(vec)
                kwargs = {"pretrainedVectors": vec, "dim": kv.vector_size}
            else:
                kwargs = {"dim": self.dim}
            fasttext.FastText.eprint = lambda *a, **k: None
            self.model_ = fasttext.train_supervised(
                train, epoch=self.epoch, lr=self.lr, wordNgrams=self.word_ngrams,
                minCount=1, loss="softmax", thread=1, seed=self.seed, verbose=0, **kwargs,
            )
        return self

    def predict_proba(self, X):
        labels, probs = self.model_.predict([self._clean(t) for t in X], k=len(self.classes_))
        idx = {c: i for i, c in enumerate(self.classes_)}
        out = np.zeros((len(X), len(self.classes_)))
        for r, (ls, ps) in enumerate(zip(labels, probs)):
            for lab, p in zip(ls, ps):
                out[r, idx[lab.removeprefix("__label__")]] = p
        return out / out.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    def size_kb(self) -> float:
        with tempfile.NamedTemporaryFile(suffix=".bin") as f:
            self.model_.save_model(f.name)
            return os.path.getsize(f.name) / 1024


# ------------------------------------------------------------------ PyTorch models
class KimCNN(nn.Module):
    """Kim (2014) CNN: parallel 1-D convolutions of width 3/4/5, max-over-time pooling."""

    def __init__(self, emb: torch.Tensor, n_classes: int, filters=100, widths=(3, 4, 5), dropout=0.5):
        super().__init__()
        self.emb = nn.Embedding.from_pretrained(emb, freeze=False, padding_idx=0)
        self.convs = nn.ModuleList(nn.Conv1d(emb.shape[1], filters, w, padding=w // 2) for w in widths)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(filters * len(widths), n_classes)

    def forward(self, ids):
        x = self.emb(ids).transpose(1, 2)
        mask = (ids == 0).unsqueeze(1)
        pooled = [F.relu(c(x))[..., : ids.shape[1]].masked_fill(mask, -1e4).amax(dim=2) for c in self.convs]
        return self.out(self.drop(torch.cat(pooled, dim=1)))


class BiLSTMAttention(nn.Module):
    """Bidirectional LSTM with additive attention pooling."""

    def __init__(self, emb: torch.Tensor, n_classes: int, hidden=128, dropout=0.5):
        super().__init__()
        self.emb = nn.Embedding.from_pretrained(emb, freeze=False, padding_idx=0)
        self.emb_drop = nn.Dropout(0.25)
        self.lstm = nn.LSTM(emb.shape[1], hidden, batch_first=True, bidirectional=True)
        self.att = nn.Linear(2 * hidden, 1)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(2 * hidden, n_classes)

    def forward(self, ids):
        lengths = (ids != 0).sum(1).clamp(min=1).cpu()
        x = self.emb_drop(self.emb(ids))
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        h, _ = self.lstm(packed)
        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True, total_length=ids.shape[1])
        scores = self.att(torch.tanh(h)).squeeze(-1).masked_fill(ids == 0, -1e4)
        ctx = (torch.softmax(scores, dim=1).unsqueeze(-1) * h).sum(1)
        return self.out(self.drop(ctx))


class SmallTransformer(nn.Module):
    """Transformer encoder trained from scratch (GloVe-initialised embeddings), mean pooled."""

    def __init__(self, emb: torch.Tensor, n_classes: int, d_model=128, layers=2, heads=4, max_len=512, dropout=0.2):
        super().__init__()
        self.emb = nn.Embedding.from_pretrained(emb, freeze=False, padding_idx=0)
        self.proj = nn.Linear(emb.shape[1], d_model)
        self.pos = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(d_model, heads, 4 * d_model, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, n_classes)

    def forward(self, ids):
        pad = ids == 0
        pos = torch.arange(ids.shape[1], device=ids.device)
        h = self.enc(self.proj(self.emb(ids)) + self.pos(pos), src_key_padding_mask=pad)
        keep = (~pad).unsqueeze(-1).float()
        return self.out(self.drop((h * keep).sum(1) / keep.sum(1).clamp(min=1)))


class MLP(nn.Module):
    """Feed-forward network (ANN) over TF-IDF features."""

    def __init__(self, n_in: int, n_classes: int, hidden=512, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(nn.Dropout(0.2), nn.Linear(n_in, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, n_classes))

    def forward(self, x):
        return self.net(x)


def _fit_torch(model, batches_fn, val_fn, epochs, lr, patience, weight_decay=1e-4):
    """Train with AdamW; keep the weights with the lowest validation loss."""
    model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best, best_state, bad = math.inf, None, 0
    for _ in range(epochs):
        model.train()
        for xb, yb in batches_fn():
            opt.zero_grad()
            F.cross_entropy(model(xb.to(DEVICE)), yb.to(DEVICE)).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        val = val_fn()
        if val < best - 1e-4:
            best, bad = val, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model


class TorchTextClassifier(ClassifierMixin, BaseEstimator):
    """Sequence models (cnn | bilstm | transformer) over GloVe-initialised token embeddings.

    10% of the training fold is held out for early stopping; the test fold is never seen.
    """

    def __init__(self, arch="cnn", max_len=256, epochs=25, lr=1e-3, batch_size=32, patience=3, seed=0):
        self.arch = arch
        self.max_len = max_len
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.patience = patience
        self.seed = seed

    def _encode(self, texts):
        ids = np.zeros((len(texts), self.max_len), dtype=np.int64)
        for i, t in enumerate(texts):
            toks = [self.vocab_.get(w, 1) for w in tokenize(t)][: self.max_len]
            ids[i, : len(toks)] = toks
        return torch.from_numpy(ids)

    @staticmethod
    def _trim(ids):
        return ids[:, : max(int((ids != 0).sum(1).max()), 5)]

    def fit(self, X, y):
        seed_all(self.seed)
        X, y = list(X), np.asarray(y)
        self.classes_ = np.unique(y)
        y_idx = np.searchsorted(self.classes_, y)
        tr, va = train_test_split(np.arange(len(X)), test_size=0.1, stratify=y_idx, random_state=self.seed)

        counts: dict[str, int] = {}
        for i in tr:
            for w in tokenize(X[i]):
                counts[w] = counts.get(w, 0) + 1
        kv = glove()
        words = [w for w, c in counts.items() if c >= 2 or w in kv.key_to_index]
        self.vocab_ = {w: i + 2 for i, w in enumerate(words)}  # 0 = pad, 1 = unknown
        emb = np.random.default_rng(self.seed).normal(0, 0.1, (len(words) + 2, kv.vector_size)).astype(np.float32)
        emb[0] = 0
        for w, i in self.vocab_.items():
            if w in kv.key_to_index:
                emb[i] = kv[w]
        emb = torch.from_numpy(emb)

        arch = {"cnn": KimCNN, "bilstm": BiLSTMAttention, "transformer": SmallTransformer}[self.arch]
        self.model_ = arch(emb, len(self.classes_))
        lr = self.lr if self.arch != "transformer" else min(self.lr, 5e-4)

        ids, yt = self._encode(X), torch.from_numpy(y_idx)
        tr_t = torch.from_numpy(tr)

        def batches():
            perm = tr_t[torch.randperm(len(tr_t))]
            for s in range(0, len(perm), self.batch_size):
                b = perm[s : s + self.batch_size]
                yield self._trim(ids[b]), yt[b]

        def val_loss():
            self.model_.eval()
            with torch.no_grad():
                logits = self.model_(self._trim(ids[va]).to(DEVICE))
            return F.cross_entropy(logits, yt[va].to(DEVICE)).item()

        _fit_torch(self.model_, batches, val_loss, self.epochs, lr, self.patience)
        return self

    def predict_proba(self, X):
        self.model_.eval()
        ids = self._encode(list(X))
        out = []
        with torch.no_grad():
            for s in range(0, len(ids), 128):
                out.append(torch.softmax(self.model_(self._trim(ids[s : s + 128]).to(DEVICE)), 1).cpu())
        return torch.cat(out).numpy()

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    def size_kb(self) -> float:
        return _params_kb(self.model_)


class TfidfMLPClassifier(ClassifierMixin, BaseEstimator):
    """ANN: TF-IDF (word 1-2 grams) -> 512-unit ReLU layer -> softmax, early stopped."""

    def __init__(self, hidden=512, max_features=30000, epochs=30, lr=1e-3, batch_size=32, patience=3, seed=0):
        self.hidden = hidden
        self.max_features = max_features
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.patience = patience
        self.seed = seed

    def fit(self, X, y):
        seed_all(self.seed)
        X, y = list(X), np.asarray(y)
        self.classes_ = np.unique(y)
        y_idx = np.searchsorted(self.classes_, y)
        tr, va = train_test_split(np.arange(len(X)), test_size=0.1, stratify=y_idx, random_state=self.seed)
        self.vec_ = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_features=self.max_features)
        self.vec_.fit([X[i] for i in tr])
        feats = torch.from_numpy(self.vec_.transform(X).toarray().astype(np.float32))
        yt = torch.from_numpy(y_idx)
        self.model_ = MLP(feats.shape[1], len(self.classes_), self.hidden)

        def batches():
            perm = torch.from_numpy(tr)[torch.randperm(len(tr))]
            for s in range(0, len(perm), self.batch_size):
                b = perm[s : s + self.batch_size]
                yield feats[b], yt[b]

        def val_loss():
            self.model_.eval()
            with torch.no_grad():
                return F.cross_entropy(self.model_(feats[va].to(DEVICE)), yt[va].to(DEVICE)).item()

        _fit_torch(self.model_, batches, val_loss, self.epochs, self.lr, self.patience, weight_decay=1e-3)
        return self

    def predict_proba(self, X):
        self.model_.eval()
        with torch.no_grad():
            x = torch.from_numpy(self.vec_.transform(list(X)).toarray().astype(np.float32)).to(DEVICE)
            return torch.softmax(self.model_(x), 1).cpu().numpy()

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    def size_kb(self) -> float:
        return _params_kb(self.model_)


# ------------------------------------------------------------------ pretrained transformers
_ENCODERS: dict = {}


class SentenceEncoder(TransformerMixin, BaseEstimator):
    """Frozen pretrained sentence encoder (sentence-transformers); embeddings cached per text."""

    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name

    def _model(self):
        if self.model_name not in _ENCODERS:
            from sentence_transformers import SentenceTransformer

            _ENCODERS[self.model_name] = (SentenceTransformer(self.model_name, device=str(DEVICE)), {})
        return _ENCODERS[self.model_name]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        model, cache = self._model()
        todo = [t for t in dict.fromkeys(X) if t not in cache]
        if todo:
            vecs = model.encode(todo, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
            cache.update(zip(todo, vecs))
        return np.vstack([cache[t] for t in X])

    def size_kb(self) -> float:
        return _params_kb(self._model()[0])

    def __getstate__(self):
        return {"model_name": self.model_name}


class HFFineTuneClassifier(ClassifierMixin, BaseEstimator):
    """Fine-tune a pretrained Hugging Face encoder (BERT, DistilBERT, RoBERTa, ...) end to end.

    AdamW, linear warm-up/decay schedule, length-bucketed batches with dynamic padding.
    This is the PyTorch equivalent of the original ktrain `text_classifier('bert')` +
    `fit_onecycle(2e-5, 4)` setup, but with max_length 256 instead of 32.
    """

    def __init__(self, model_name="bert-base-uncased", max_len=256, epochs=3, lr=3e-5, batch_size=16,
                 warmup=0.1, weight_decay=0.01, seed=0):
        self.model_name = model_name
        self.max_len = max_len
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.warmup = warmup
        self.weight_decay = weight_decay
        self.seed = seed

    def _batches(self, enc, order):
        for s in range(0, len(order), self.batch_size):
            b = order[s : s + self.batch_size]
            yield b, self.tok_.pad([{"input_ids": enc[i]} for i in b], return_tensors="pt")

    def fit(self, X, y):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
        seed_all(self.seed)
        X, y = list(X), np.asarray(y)
        self.classes_ = np.unique(y)
        yt = torch.from_numpy(np.searchsorted(self.classes_, y))
        self.tok_ = AutoTokenizer.from_pretrained(self.model_name)
        self.model_ = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=len(self.classes_)
        ).to(DEVICE)
        enc = self.tok_(X, truncation=True, max_length=self.max_len)["input_ids"]

        no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
        groups = [
            {"params": [p for n, p in self.model_.named_parameters() if not n.endswith(no_decay)], "weight_decay": self.weight_decay},
            {"params": [p for n, p in self.model_.named_parameters() if n.endswith(no_decay)], "weight_decay": 0.0},
        ]
        opt = torch.optim.AdamW(groups, lr=self.lr)
        steps = self.epochs * math.ceil(len(X) / self.batch_size)
        sched = get_linear_schedule_with_warmup(opt, int(self.warmup * steps), steps)
        rng = np.random.default_rng(self.seed)
        lengths = np.array([len(e) for e in enc])

        for _ in range(self.epochs):
            self.model_.train()
            # Bucket by length (with noise) so batches pad little, then shuffle batch order.
            order = np.argsort(lengths + rng.uniform(0, 20, len(lengths)))
            chunks = [order[s : s + self.batch_size] for s in range(0, len(order), self.batch_size)]
            rng.shuffle(chunks)
            for b in chunks:
                batch = self.tok_.pad([{"input_ids": enc[i]} for i in b], return_tensors="pt").to(DEVICE)
                loss = self.model_(**batch, labels=yt[b].to(DEVICE)).loss
                loss.backward()
                nn.utils.clip_grad_norm_(self.model_.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
        return self

    def predict_proba(self, X):
        self.model_.eval()
        X = list(X)
        enc = self.tok_(X, truncation=True, max_length=self.max_len)["input_ids"]
        order = np.argsort([len(e) for e in enc])
        out = np.zeros((len(X), len(self.classes_)), dtype=np.float32)
        with torch.no_grad():
            for b, batch in self._batches(enc, order):
                out[b] = torch.softmax(self.model_(**batch.to(DEVICE)).logits, 1).cpu().numpy()
        return out

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    def size_kb(self) -> float:
        return _params_kb(self.model_)
