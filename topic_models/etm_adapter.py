"""ETM adapter — LM-grounded, retriever-space topic embeddings.

Faithful reimplementation of the Embedded Topic Model (Dieng et al.; cloned repo
``./ETM/etm.py``) with one defining change: the word embeddings ``rho`` are the
**retriever LM's embeddings of the vocabulary words** (frozen). Because the topic
embeddings ``alphas.weight`` (``alpha_k in R^d``) live in the same space as
``rho``, and ``rho`` is the retriever space, the learned ``alpha_k`` are *native
centroids* in the retriever space — no empirical step required.

ETM identity is preserved with ``theta_input="bow"`` and ``train_embeddings=False``.
Setting ``theta_input="lm_embedding"`` feeds the document's LM embedding to the
variational encoder (a ZeroShotTM-style variant); ``"concat"`` concatenates BoW and
the LM embedding (a CombinedTM-style variant). All three modes are implemented.
"""

from __future__ import annotations

import json
import os
import random
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from topic_models.base import TopicModel, TopicTrainConfig
from topic_models.registry import register_topic_model
from topic_models.wikiweb2m import tokenize

if TYPE_CHECKING:
    from src.encoders.retriever_encoder import RetrieverEncoder
    from topic_models.wikiweb2m import TopicCorpus


_ACTS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "softplus": nn.Softplus,
    "sigmoid": nn.Sigmoid,
    "leakyrelu": nn.LeakyReLU,
    "elu": nn.ELU,
    "selu": nn.SELU,
}


def _act(name: str) -> nn.Module:
    if name not in _ACTS:
        raise ValueError(f"Unknown theta_act {name!r}; choose from {sorted(_ACTS)}.")
    return _ACTS[name]()


class _ETMCore(nn.Module):
    """Faithful ETM core with frozen LM-grounded word embeddings."""

    def __init__(
        self,
        num_topics: int,
        vocab_size: int,
        rho_size: int,
        t_hidden_size: int = 800,
        theta_act: str = "relu",
        rho_init: torch.Tensor | None = None,
        train_embeddings: bool = False,
        enc_drop: float = 0.5,
        theta_input: str = "bow",
        lm_dim: int | None = None,
    ):
        super().__init__()
        self.num_topics = int(num_topics)
        self.vocab_size = int(vocab_size)
        self.rho_size = int(rho_size)
        self.theta_input = theta_input
        self.train_embeddings = bool(train_embeddings)
        self.enc_drop = float(enc_drop)
        self.lm_dim = int(lm_dim) if lm_dim is not None else int(rho_size)

        # Word embeddings rho (V, rho_size). Trainable Linear or frozen buffer.
        if train_embeddings:
            self.rho = nn.Linear(rho_size, vocab_size, bias=False)  # rows = word emb
            if rho_init is not None:
                with torch.no_grad():
                    self.rho.weight.copy_(rho_init)
        else:
            if rho_init is None:
                rho_init = torch.randn(vocab_size, rho_size)
            self.register_buffer("rho", rho_init.clone().float())

        # Topic embeddings alpha_k = alphas.weight[k] in R^{rho_size}.
        self.alphas = nn.Linear(rho_size, num_topics, bias=False)

        # Variational encoder q(theta | input).
        if theta_input == "bow":
            in_dim = vocab_size
        elif theta_input == "lm_embedding":
            in_dim = self.lm_dim
        elif theta_input == "concat":
            in_dim = vocab_size + self.lm_dim
        else:
            raise ValueError(
                f"Unknown theta_input {theta_input!r} "
                "(expected 'bow' | 'lm_embedding' | 'concat')."
            )
        self.q_theta = nn.Sequential(
            nn.Linear(in_dim, t_hidden_size),
            _act(theta_act),
            nn.Linear(t_hidden_size, t_hidden_size),
            _act(theta_act),
        )
        self.mu_q = nn.Linear(t_hidden_size, num_topics, bias=True)
        self.logsigma_q = nn.Linear(t_hidden_size, num_topics, bias=True)
        self.t_drop = nn.Dropout(enc_drop)

    def _rho_matrix(self) -> torch.Tensor:
        return self.rho.weight if self.train_embeddings else self.rho  # (V, rho_size)

    def get_beta(self) -> torch.Tensor:
        """Topic-word distribution ``(K, V)``; each topic row sums to 1 over vocab."""
        logits = self.alphas(self._rho_matrix())  # (V, K)
        beta = F.softmax(logits, dim=0).transpose(0, 1)  # (K, V)
        return beta

    def topic_embeddings(self) -> torch.Tensor:
        """``alphas.weight`` -> ``(K, rho_size)`` (native retriever-space centroids)."""
        return self.alphas.weight  # (K, rho_size)

    def encode(self, q_input: torch.Tensor):
        q = self.q_theta(q_input)
        if self.enc_drop > 0:
            q = self.t_drop(q)
        mu = self.mu_q(q)
        logsigma = self.logsigma_q(q)
        kl = -0.5 * torch.sum(
            1 + logsigma - mu.pow(2) - logsigma.exp(), dim=-1
        ).mean()
        return mu, logsigma, kl

    def reparameterize(self, mu: torch.Tensor, logsigma: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logsigma)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def get_theta(self, q_input: torch.Tensor):
        mu, logsigma, kl = self.encode(q_input)
        z = self.reparameterize(mu, logsigma)
        theta = F.softmax(z, dim=-1)
        return theta, kl

    def forward(self, bows: torch.Tensor, q_input: torch.Tensor):
        theta, kl = self.get_theta(q_input)
        beta = self.get_beta()
        log_p = torch.log(theta @ beta + 1e-12)  # (B, V)
        recon = -(log_p * bows).sum(dim=1).mean()
        return recon, kl


@register_topic_model("etm")
class ETMTopicModel(TopicModel):
    """ETM with frozen retriever-LM word embeddings → native topic centroids."""

    name = "etm"

    def __init__(
        self,
        num_topics: int,
        encoder: "RetrieverEncoder",
        centroid_normalize: bool = True,
        device: str | None = None,
        t_hidden_size: int = 800,
        theta_act: str = "relu",
        enc_drop: float = 0.5,
        theta_input: str = "bow",
        train_embeddings: bool = False,
        **kwargs,
    ):
        super().__init__(num_topics, encoder, centroid_normalize, device)
        self.t_hidden_size = int(t_hidden_size)
        self.theta_act = theta_act
        self.enc_drop = float(enc_drop)
        self.theta_input = theta_input
        self.train_embeddings = bool(train_embeddings)

        self.vocab: list[str] | None = None
        self._w2i: dict[str, int] | None = None
        self.core: _ETMCore | None = None
        self.train_losses: list[float] = []

    # ---- interface ----
    @property
    def supports_native_centroids(self) -> bool:
        return True

    @torch.no_grad()
    def _embed_vocab(self, vocab: list[str], batch_size: int = 256) -> torch.Tensor:
        """Frozen LM-grounding: ``rho = L2norm(encoder.encode(vocab))`` -> ``(V, d)``."""
        rho = self.encoder.encode(vocab, is_query=False, batch_size=batch_size)
        rho = F.normalize(rho.float(), p=2, dim=-1)
        return rho.to(self.device)

    def _build_core(self, vocab: list[str]) -> None:
        self.vocab = list(vocab)
        self._w2i = {w: i for i, w in enumerate(self.vocab)}
        rho = self._embed_vocab(self.vocab)  # (V, d), frozen
        self.core = _ETMCore(
            num_topics=self.num_topics,
            vocab_size=len(self.vocab),
            rho_size=self.embedding_dim,
            t_hidden_size=self.t_hidden_size,
            theta_act=self.theta_act,
            rho_init=rho,
            train_embeddings=self.train_embeddings,
            enc_drop=self.enc_drop,
            theta_input=self.theta_input,
            lm_dim=self.embedding_dim,
        ).to(self.device)

    @torch.no_grad()
    def _bow(self, texts: list[str]) -> torch.Tensor:
        if self._w2i is None or self.vocab is None:
            raise RuntimeError("ETM vocab not built; call fit(...) first.")
        V = len(self.vocab)
        out = torch.zeros(len(texts), V, dtype=torch.float32)
        for b, text in enumerate(texts):
            for tok in tokenize(text):
                idx = self._w2i.get(tok)
                if idx is not None:
                    out[b, idx] += 1.0
            s = out[b].sum()
            if s > 0:
                out[b] /= s
        return out.to(self.device)

    @torch.no_grad()
    def _q_input(self, texts: list[str]) -> torch.Tensor:
        if self.theta_input == "bow":
            return self._bow(texts)
        if self.theta_input == "lm_embedding":
            return self.encoder.encode(texts, is_query=False).float().to(self.device)
        if self.theta_input == "concat":
            bow = self._bow(texts)
            lm = self.encoder.encode(texts, is_query=False).float().to(self.device)
            return torch.cat([bow, lm], dim=-1)
        raise ValueError(f"Unknown theta_input {self.theta_input!r}.")

    @torch.no_grad()
    def encode_topic_distribution(
        self, texts: list[str], batch_size: int = 8
    ) -> torch.Tensor:
        if self.core is None:
            raise RuntimeError("ETM core not built; call fit(...) first.")
        self.core.eval()
        rows: list[torch.Tensor] = []
        for start in range(0, len(texts), batch_size):
            sub = texts[start : start + batch_size]
            q = self._q_input(sub)
            theta, _ = self.core.get_theta(q)
            rows.append(theta.float())
        if not rows:
            return torch.empty(0, self.num_topics, device=self.device, dtype=torch.float32)
        return torch.cat(rows, dim=0)

    def _native_centroids(self) -> torch.Tensor:
        if self.core is None:
            raise RuntimeError("ETM core not built; call fit(...) first.")
        return self.core.topic_embeddings()  # (K, d), un-normalized

    def fit(self, corpus: "TopicCorpus", cfg: "TopicTrainConfig") -> None:
        vocab = corpus.vocab or corpus.build_vocab(cfg.vocab_size, cfg.min_word_freq)
        self._build_core(vocab)
        assert self.core is not None
        # Snapshot initial topic embeddings so callers can confirm training moved them.
        self._init_topic_embeddings = self.core.topic_embeddings().detach().clone()

        # Adam over all core parameters. When train_embeddings is False, rho is a
        # frozen buffer (not a parameter) so it is automatically excluded.
        optimizer = torch.optim.Adam(self.core.parameters(), lr=cfg.lr)
        rng = random.Random(cfg.seed)
        torch.manual_seed(cfg.seed)

        docs = list(corpus.documents)
        self.train_losses = []
        self.core.train()
        for epoch in range(cfg.epochs):
            order = list(range(len(docs)))
            rng.shuffle(order)
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, len(order), cfg.batch_size):
                idxs = order[start : start + cfg.batch_size]
                batch = [docs[i] for i in idxs]
                bows = self._bow(batch)
                q = self._q_input(batch)
                recon, kl = self.core(bows, q)
                loss = recon + kl
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.detach())
                n_batches += 1
            avg = epoch_loss / max(n_batches, 1)
            self.train_losses.append(avg)
            print(f"[etm] epoch {epoch}: loss={avg:.4f}")

        self.core.eval()
        self.set_centroid_source(cfg.centroid_source)

    # ---- persistence ----
    def _save_state(self, out_dir: str) -> None:
        if self.core is None:
            raise RuntimeError("Nothing to save; ETM core not built.")
        torch.save(self.core.state_dict(), os.path.join(out_dir, "etm_core.pt"))
        with open(os.path.join(out_dir, "etm_meta.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "vocab": self.vocab,
                    "theta_input": self.theta_input,
                    "t_hidden_size": self.t_hidden_size,
                    "theta_act": self.theta_act,
                    "enc_drop": self.enc_drop,
                    "train_embeddings": self.train_embeddings,
                },
                f,
            )

    def _load_state(self, in_dir: str) -> None:
        with open(os.path.join(in_dir, "etm_meta.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.theta_input = meta["theta_input"]
        self.t_hidden_size = int(meta["t_hidden_size"])
        self.theta_act = meta["theta_act"]
        self.enc_drop = float(meta["enc_drop"])
        self.train_embeddings = bool(meta["train_embeddings"])
        self._build_core(meta["vocab"])
        assert self.core is not None
        sd = torch.load(os.path.join(in_dir, "etm_core.pt"), map_location="cpu", weights_only=False)
        self.core.load_state_dict(sd, strict=True)
        self.core.to(self.device)
        self.core.eval()
