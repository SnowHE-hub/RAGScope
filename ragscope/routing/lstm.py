"""A small CPU LSTM classifier; torch is imported only for real fit/load."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re
from time import perf_counter

import numpy as np

from ragscope.indexing.identity import artifact, atomic_json, signature, verify_artifacts
from .core import ROUTES, RouteDecision, RoutingRequest, validate_training_splits, validate_proxy_provenance


def words(text):
    return re.findall(r"[a-z0-9_]+|[\u3400-\u9fff]", text.casefold())


def parameter_budget(vocabulary_size, embedding_size=64, hidden_size=64):
    return vocabulary_size * embedding_size + 4 * hidden_size * (embedding_size + hidden_size) + 8 * hidden_size + hidden_size * 3 + 3


def _network(vocabulary_size, embedding_size, hidden_size):
    import torch
    from torch import nn
    class Network(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(vocabulary_size, embedding_size, padding_idx=0)
            self.lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True)
            self.output = nn.Linear(hidden_size, len(ROUTES))
        def forward(self, tokens, lengths):
            embedded = self.embedding(tokens)
            packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
            _, (hidden, _) = self.lstm(packed)
            return self.output(hidden[-1])
    return Network()


class LSTMRouter:
    def __init__(self, *, max_vocabulary=8000, embedding_size=64, hidden_size=64,
                 max_tokens=128, max_epochs=20, patience=3, seed=42):
        if max_vocabulary < 2 or max_tokens < 1 or not 1 <= max_epochs <= 20 or not 1 <= patience <= 3:
            raise ValueError("Invalid bounded LSTM configuration")
        if min(embedding_size, hidden_size) < 1 or parameter_budget(max_vocabulary, embedding_size, hidden_size) >= 2_000_000:
            raise ValueError("LSTM routing model must contain fewer than 2M parameters")
        self.config = {"max_vocabulary": max_vocabulary, "embedding_size": embedding_size,
                       "hidden_size": hidden_size, "max_tokens": max_tokens,
                       "max_epochs": max_epochs, "patience": patience, "seed": seed}
        self.vocabulary, self.model, self.constant, self.report = {}, None, None, {}

    def _encode(self, text):
        tokens = words(text)
        ids = [self.vocabulary.get(token, 1) for token in tokens[:self.config["max_tokens"]]] or [1]
        return ids, max(0, len(tokens) - self.config["max_tokens"])

    def _batch(self, rows):
        import torch
        encoded = [self._encode(RoutingRequest.from_query(row).features_text())[0] for row in rows]
        lengths = torch.tensor([len(ids) for ids in encoded], dtype=torch.long)
        padded = torch.zeros((len(rows), int(lengths.max())), dtype=torch.long)
        for i, ids in enumerate(encoded):
            padded[i, :len(ids)] = torch.tensor(ids)
        return padded, lengths

    def fit(self, train, dev, *, data_kind="real_generation_proxy"):
        counts = validate_training_splits(train, dev)
        validate_proxy_provenance(train, dev, data_kind)
        self.report = {**counts, "config": self.config, "data_kind": data_kind, "history": [],
                       "train_query_ids": [row["query_id"] for row in train], "dev_query_ids": [row["query_id"] for row in dev]}
        if counts["single_class"]:
            self.constant = train[0]["label"]
            self.report.update(status="PASS_FALLBACK_SINGLE_CLASS", parameter_count=0)
            return self
        import torch
        started = perf_counter()
        torch.set_num_threads(4)
        torch.manual_seed(self.config["seed"])
        counts = Counter(token for row in train for token in words(RoutingRequest.from_query(row).features_text()))
        ordered = sorted(counts, key=lambda token: (-counts[token], token))[:self.config["max_vocabulary"] - 2]
        self.vocabulary = {token: i + 2 for i, token in enumerate(ordered)}
        self.model = _network(len(self.vocabulary) + 2, self.config["embedding_size"], self.config["hidden_size"]).cpu()
        n_parameters = sum(parameter.numel() for parameter in self.model.parameters())
        if n_parameters >= 2_000_000:
            raise ValueError("Actual LSTM model exceeds parameter limit")
        optimizer = torch.optim.Adam(self.model.parameters(), lr=.001)
        labels = torch.tensor([ROUTES.index(row["label"]) for row in train], dtype=torch.long)
        dev_labels = torch.tensor([ROUTES.index(row["label"]) for row in dev], dtype=torch.long)
        dev_x, dev_lengths = self._batch(dev)
        best, best_loss, stale = None, float("inf"), 0
        rng = np.random.default_rng(self.config["seed"])
        for epoch in range(self.config["max_epochs"]):
            self.model.train()
            order = rng.permutation(len(train))
            for offset in range(0, len(train), 16):
                indices = order[offset:offset + 16]
                tokens, lengths = self._batch([train[int(i)] for i in indices])
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.cross_entropy(self.model(tokens, lengths), labels[indices.tolist()])
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite LSTM training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
            self.model.eval()
            with torch.no_grad():
                logits = self.model(dev_x, dev_lengths)
                dev_loss = float(torch.nn.functional.cross_entropy(logits, dev_labels))
                accuracy = float((logits.argmax(-1) == dev_labels).float().mean())
            self.report["history"].append({"epoch": epoch + 1, "dev_log_loss": dev_loss, "dev_accuracy": accuracy})
            if dev_loss < best_loss - 1e-8:
                best = {name: tensor.detach().cpu().clone() for name, tensor in self.model.state_dict().items()}
                best_loss, stale = dev_loss, 0
            else:
                stale += 1
            if stale >= self.config["patience"]:
                break
        self.model.load_state_dict(best)
        self.model.eval()
        self.report.update(status="TRAINED", parameter_count=n_parameters, training_elapsed_s=perf_counter() - started,
                           truncated_train_queries=sum(self._encode(RoutingRequest.from_query(row).features_text())[1] > 0 for row in train),
                           truncated_dev_queries=sum(self._encode(RoutingRequest.from_query(row).features_text())[1] > 0 for row in dev))
        return self

    def choose(self, request):
        started = perf_counter()
        if self.constant is not None:
            return RouteDecision(self.constant, "lstm_single_class_fallback", elapsed_s=perf_counter() - started)
        if self.model is None:
            raise ValueError("LSTM router has not been trained")
        import torch
        ids, truncated = self._encode(request.features_text())
        with torch.no_grad():
            logits = self.model(torch.tensor([ids]), torch.tensor([len(ids)]))
            probabilities = torch.softmax(logits, -1)[0].numpy()
        values = {route: float(value) for route, value in zip(ROUTES, probabilities)}
        return RouteDecision(max(values, key=values.get), "small_lstm", values,
                             elapsed_s=perf_counter() - started, metadata={"truncated_router_tokens": truncated})

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "weights.npz"
        if path.exists() or (directory / "manifest.json").exists():
            raise FileExistsError("Router artifacts are immutable")
        if not self.report:
            raise ValueError("Cannot save untrained router")
        weights = {key: value.detach().cpu().numpy() for key, value in self.model.state_dict().items()} if self.model else {}
        with path.open("wb") as handle:
            np.savez(handle, **weights)
        state = {"config": self.config, "vocabulary": self.vocabulary, "constant": self.constant, "report": self.report}
        atomic_json(directory / "manifest.json", {"identity": state, "signature": signature(state), "artifacts": [artifact(path)]})

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        state = manifest["identity"]
        if signature(state) != manifest["signature"] or not verify_artifacts(manifest["artifacts"]):
            raise ValueError("LSTM router artifact integrity failure")
        router = cls(**state["config"])
        router.vocabulary, router.constant, router.report = state["vocabulary"], state["constant"], state["report"]
        if router.constant is None:
            import torch
            router.model = _network(len(router.vocabulary) + 2, router.config["embedding_size"], router.config["hidden_size"])
            with np.load(directory / "weights.npz", allow_pickle=False) as weights:
                router.model.load_state_dict({key: torch.from_numpy(weights[key]) for key in weights.files}, strict=True)
            router.model.eval()
        return router
