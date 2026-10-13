"""Bounded TF-IDF logistic/MLP training with JSON+npz inference artifacts."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
from time import perf_counter
import warnings

import numpy as np

from ragscope.indexing.identity import artifact, atomic_json, signature, verify_artifacts
from .core import ROUTES, RouteDecision, RoutingRequest, validate_training_splits, validate_proxy_provenance


def _softmax(values):
    values = np.asarray(values, dtype=float)
    values = np.exp(values - np.max(values, axis=-1, keepdims=True))
    return values / values.sum(axis=-1, keepdims=True)


def _classification_report(probabilities, classes, labels):
    predictions = [classes[int(index)] for index in np.argmax(probabilities, axis=1)]
    positions = {name: i for i, name in enumerate(classes)}
    losses = [-np.log(max(1e-12, probabilities[i, positions[label]])) if label in positions else -np.log(1e-12)
              for i, label in enumerate(labels)]
    return {"accuracy": float(np.mean([a == b for a, b in zip(predictions, labels)])),
            "log_loss": float(np.mean(losses)), "predicted_counts": dict(Counter(predictions))}


class TfidfRouter:
    def __init__(self, kind="linear", *, max_features=2048, reduction_components=64,
                 hidden_units=32, max_epochs=20, patience=3, seed=42):
        if kind not in {"linear", "mlp"} or not 1 <= max_epochs <= 20 or not 1 <= patience <= 3:
            raise ValueError("Use linear/mlp, <=20 epochs and patience <=3")
        if not 1 <= max_features <= 8192 or not 1 <= hidden_units <= 128 or not 1 <= reduction_components <= 128:
            raise ValueError("Router feature/network budget exceeded")
        self.config = {"kind": kind, "max_features": max_features, "reduction_components": reduction_components,
                       "hidden_units": hidden_units, "max_epochs": max_epochs, "patience": patience, "seed": seed}
        self.kind, self.seed = kind, seed
        self.vectorizer = None
        self.weights = {}
        self.classes = list(ROUTES)
        self.constant = None
        self.report = {}

    def fit(self, train, dev, *, data_kind="real_generation_proxy"):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.neural_network import MLPClassifier
        from sklearn.decomposition import TruncatedSVD

        counts = validate_training_splits(train, dev)
        validate_proxy_provenance(train, dev, data_kind)
        started = perf_counter()
        self.report = {**counts, "data_kind": data_kind, "config": self.config, "history": [], "warnings": []}
        self.report.update(train_query_ids=[row["query_id"] for row in train], dev_query_ids=[row["query_id"] for row in dev])
        y = np.asarray([row["label"] for row in train])
        dev_y = [row["label"] for row in dev]
        if counts["single_class"]:
            self.constant = str(y[0])
            self.report.update(status="PASS_FALLBACK_SINGLE_CLASS", constant=self.constant,
                               training_elapsed_s=perf_counter() - started)
            return self
        self.vectorizer = TfidfVectorizer(max_features=self.config["max_features"], ngram_range=(1, 2),
                                          sublinear_tf=True, token_pattern=r"(?u)\b\w+\b")
        x = self.vectorizer.fit_transform([RoutingRequest.from_query(row).features_text() for row in train])
        dev_x = self.vectorizer.transform([RoutingRequest.from_query(row).features_text() for row in dev])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if self.kind == "linear":
                model = LogisticRegression(max_iter=self.config["max_epochs"], random_state=self.seed, C=1.0)
                model.fit(x, y)
                self.weights = {"coef": model.coef_, "intercept": model.intercept_}
                self.classes = model.classes_.tolist()
                self.report["history"].append({"solver_iterations": int(np.max(model.n_iter_)),
                                               **_classification_report(model.predict_proba(dev_x), self.classes, dev_y)})
            else:
                dimension = min(self.config["reduction_components"], x.shape[0] - 1, x.shape[1] - 1)
                if dimension < 1:
                    raise ValueError("Insufficient train features for a reduced TF-IDF MLP")
                reduction = TruncatedSVD(n_components=dimension, random_state=self.seed)
                x = reduction.fit_transform(x)
                dev_x = reduction.transform(dev_x)
                model = MLPClassifier(hidden_layer_sizes=(self.config["hidden_units"],), random_state=self.seed,
                                      max_iter=1, batch_size=min(16, len(train)), shuffle=True,
                                      solver="adam", learning_rate_init=.001)
                best, best_loss, stale = None, float("inf"), 0
                for epoch in range(self.config["max_epochs"]):
                    model.partial_fit(x, y, classes=np.asarray(ROUTES))
                    score = _classification_report(model.predict_proba(dev_x), model.classes_, dev_y)
                    self.report["history"].append({"epoch": epoch + 1, **score})
                    if score["log_loss"] < best_loss - 1e-8:
                        best, best_loss, stale = deepcopy(model), score["log_loss"], 0
                    else:
                        stale += 1
                    if stale >= self.config["patience"]:
                        break
                self.classes = best.classes_.tolist()
                self.weights = {"svd": reduction.components_, "coef0": best.coefs_[0], "coef1": best.coefs_[1],
                                "intercept0": best.intercepts_[0], "intercept1": best.intercepts_[1]}
            self.report["warnings"] = [str(warning.message) for warning in caught]
        self.report.update(status="TRAINED", training_elapsed_s=perf_counter() - started,
                           n_features=x.shape[1], train_query_ids=[row["query_id"] for row in train],
                           dev_query_ids=[row["query_id"] for row in dev])
        return self

    def _predict(self, texts):
        if self.constant is not None:
            return np.asarray([[float(route == self.constant) for route in self.classes] for _ in texts])
        if self.vectorizer is None or not self.weights:
            raise ValueError("Router has not been trained")
        features = self.vectorizer.transform(texts)
        if self.kind == "linear":
            logits = np.asarray(features @ self.weights["coef"].T + self.weights["intercept"])
            if logits.shape[1] == 1:
                positive = 1.0 / (1.0 + np.exp(-np.clip(logits[:, 0], -700, 700)))
                return np.column_stack([1 - positive, positive])
            return _softmax(logits)
        projected = np.asarray(features @ self.weights["svd"].T)
        hidden = np.maximum(0, projected @ self.weights["coef0"] + self.weights["intercept0"])
        return _softmax(hidden @ self.weights["coef1"] + self.weights["intercept1"])

    def choose(self, request):
        started = perf_counter()
        probabilities = self._predict([request.features_text()])[0]
        values = {route: 0.0 for route in ROUTES}
        values.update({route: float(value) for route, value in zip(self.classes, probabilities)})
        selected = max(ROUTES, key=lambda route: values[route])
        return RouteDecision(selected, f"tfidf_{self.kind}", values, elapsed_s=perf_counter() - started,
                             metadata={"constant_single_class": self.constant is not None})

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        if not self.report:
            raise ValueError("Cannot save an untrained router")
        arrays = dict(self.weights)
        if self.vectorizer is not None:
            arrays["idf"] = self.vectorizer.idf_
        path = directory / "weights.npz"
        if path.exists() or (directory / "manifest.json").exists():
            raise FileExistsError("Router artifacts are immutable; choose a new run directory")
        with path.open("wb") as handle:
            np.savez(handle, **arrays)
        identity = {"config": self.config, "report": self.report, "classes": self.classes,
                    "constant": self.constant,
                    "vocabulary": {key: int(value) for key, value in self.vectorizer.vocabulary_.items()} if self.vectorizer is not None else None}
        atomic_json(directory / "manifest.json", {"identity": identity, "signature": signature(identity),
                                                   "artifacts": [artifact(path)]})

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        state = manifest["identity"]
        if signature(state) != manifest["signature"] or not verify_artifacts(manifest["artifacts"]):
            raise ValueError("Router artifact integrity check failed")
        router = cls(**state["config"])
        router.classes, router.constant, router.report = state["classes"], state["constant"], state["report"]
        with np.load(directory / "weights.npz", allow_pickle=False) as arrays:
            router.weights = {key: arrays[key] for key in arrays.files if key != "idf"}
            if state["vocabulary"] is not None:
                from sklearn.feature_extraction.text import TfidfVectorizer
                router.vectorizer = TfidfVectorizer(vocabulary=state["vocabulary"], ngram_range=(1, 2),
                                                    sublinear_tf=True, token_pattern=r"(?u)\b\w+\b")
                router.vectorizer.idf_ = arrays["idf"]
        return router
