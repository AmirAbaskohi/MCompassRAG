"""SoftLTM adapter — soft-label ProdLDA with retriever-space topic embeddings.

Borrows the soft-label recipe from SoftLTM (arXiv:2602.17907) and extends it so
topics are embeddings in the **retriever** space:

* A generative **label LM** (``AutoModelForCausalLM``, default
  ``meta-llama/Llama-3.2-1B-Instruct``) produces, for each document, (a) the soft
  target — a temperature-scaled softmax over the label LM's next-token logits
  restricted to the fixed vocabulary words — and (b) the document input ``x_emb``,
  the LM's final-layer hidden state at the last (prompt-final) token.
* A **ProdLDA** encoder maps ``x_emb`` to a logistic-normal latent and ``theta``.
* An **ETM-style decoder** uses topic embeddings ``alpha = alphas.weight`` and frozen
  word embeddings ``rho`` = the *retriever* encoder's embeddings of the vocab words,
  so ``alpha_k in R^d`` are native centroids in the retriever space.

Loss: ``L = lambda * KL(p || y_target) + KL(q(z|x_emb) || prior)`` with a
logistic-normal (Srivastava & Sutton) prior. Defaults tau=3, lambda=1e3, prior_a=0.02.
The (x_emb, y_target) pass is cached so the label LM runs once, not every epoch.
"""

from __future__ import annotations

import json
import os
import random
import tempfile
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from topic_models.base import TopicModel, TopicTrainConfig
from topic_models.registry import register_topic_model

if TYPE_CHECKING:
    from src.encoders.retriever_encoder import RetrieverEncoder
    from topic_models.wikiweb2m import TopicCorpus


DEFAULT_LABEL_PROMPT = (
    "Generate a single word that best captures the main theme of the "
    "following document.\n\nDocument: {doc}\n\nTheme word:"
)


class _LabelLM(nn.Module):
    """Frozen generative LM producing soft targets + a document input embedding."""

    def __init__(
        self,
        model_name: str,
        vocab: list[str],
        prompt_template: str,
        device: str,
        dtype: torch.dtype = torch.bfloat16,
        max_length: int = 1024,
    ):
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.prompt_template = prompt_template
        self.device_ = device
        self.max_length = max_length
        self.call_count = 0  # instrumentation: counts forward() invocations

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, output_hidden_states=True, dtype=dtype
        )
        self.model.to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        # First-subword approximation: a multi-token word is represented by the id
        # of its leading subword (with a leading space, matching how the word would
        # appear mid-sentence). Adequate as a vocab gather target for soft labels.
        ids = []
        for w in vocab:
            toks = self.tokenizer(" " + w, add_special_tokens=False).input_ids
            ids.append(toks[0] if len(toks) > 0 else self.tokenizer.unk_token_id or 0)
        self.register_buffer(
            "vocab_token_ids", torch.tensor(ids, dtype=torch.long), persistent=False
        )

    @property
    def hidden_dim(self) -> int:
        return int(self.model.config.hidden_size)

    @torch.no_grad()
    def forward(
        self, texts: list[str], batch_size: int = 8
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(x_emb (B,h_lm), logits_V (B,|V|))`` float32 on CPU."""
        self.call_count += 1
        safe = [t if (isinstance(t, str) and t.strip()) else " " for t in texts]
        prompts = [self.prompt_template.format(doc=t) for t in safe]

        x_chunks: list[torch.Tensor] = []
        v_chunks: list[torch.Tensor] = []
        vtid = self.vocab_token_ids.to(self.device_)
        for start in range(0, len(prompts), batch_size):
            sub = prompts[start : start + batch_size]
            enc = self.tokenizer(
                sub,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_length,
            )
            enc = {k: v.to(self.device_) for k, v in enc.items()}
            out = self.model(**enc, output_hidden_states=True)
            # Left padding -> the last position (-1) is the final real token.
            x_emb = out.hidden_states[-1][:, -1, :].float()  # (b, h_lm)
            next_logits = out.logits[:, -1, :].float()  # (b, vocab_lm)
            logits_v = next_logits.index_select(1, vtid)  # (b, |V|)
            x_chunks.append(x_emb.cpu())
            v_chunks.append(logits_v.cpu())

        if not x_chunks:
            return (
                torch.empty(0, self.hidden_dim, dtype=torch.float32),
                torch.empty(0, int(self.vocab_token_ids.numel()), dtype=torch.float32),
            )
        return torch.cat(x_chunks, 0), torch.cat(v_chunks, 0)


class _ProdLDACore(nn.Module):
    """ProdLDA encoder + ETM-grounded decoder with frozen retriever-space rho."""

    def __init__(
        self,
        num_topics: int,
        vocab_size: int,
        lm_dim: int,
        rho_size: int,
        rho_init: torch.Tensor,
        t_hidden: int = 200,
        dropout: float = 0.2,
        prior_a: float = 0.02,
    ):
        super().__init__()
        self.num_topics = int(num_topics)
        self.vocab_size = int(vocab_size)

        self.encoder = nn.Sequential(
            nn.Linear(lm_dim, t_hidden),
            nn.Softplus(),
            nn.Linear(t_hidden, t_hidden),
            nn.Softplus(),
            nn.Dropout(dropout),
        )
        self.mu_q = nn.Linear(t_hidden, num_topics)
        self.logsigma_q = nn.Linear(t_hidden, num_topics)
        self.mu_bn = nn.BatchNorm1d(num_topics, affine=False)
        self.logsigma_bn = nn.BatchNorm1d(num_topics, affine=False)

        # ETM-style decoder: topic embeddings alpha; frozen word embeddings rho.
        self.alphas = nn.Linear(rho_size, num_topics, bias=False)
        self.register_buffer("rho", rho_init.clone().float())  # (V, rho_size), frozen
        self.decoder_bn = nn.BatchNorm1d(vocab_size, affine=False)

        # Logistic-normal prior variance for a symmetric Dirichlet concentration a.
        prior_var = (1.0 / prior_a) * (1.0 - 1.0 / num_topics)
        self.register_buffer("prior_var", torch.tensor(float(prior_var)))

    def encode(self, x_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x_emb)
        mu = self.mu_bn(self.mu_q(h))
        logsigma = self.logsigma_bn(self.logsigma_q(h))
        return mu, logsigma

    def reparameterize(self, mu: torch.Tensor, logsigma: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logsigma)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def get_theta(self, x_emb: torch.Tensor):
        mu, logsigma = self.encode(x_emb)
        z = self.reparameterize(mu, logsigma)
        theta = F.softmax(z, dim=-1)
        return theta, (mu, logsigma)

    def decode(self, theta: torch.Tensor) -> torch.Tensor:
        beta = self.alphas.weight @ self.rho.t()  # (K, V)
        recon_logits = theta @ beta  # (B, V)
        recon_logits = self.decoder_bn(recon_logits)
        return F.softmax(recon_logits, dim=-1)

    def prior_kl(self, mu: torch.Tensor, logsigma: torch.Tensor) -> torch.Tensor:
        pv = self.prior_var
        var = torch.exp(logsigma)
        term = var / pv + mu.pow(2) / pv - 1.0 + torch.log(pv) - logsigma
        return 0.5 * term.sum(dim=1).mean()

    def forward(self, x_emb: torch.Tensor, y_target: torch.Tensor, lam: float):
        theta, (mu, logsigma) = self.get_theta(x_emb)
        p = self.decode(theta)
        eps = 1e-12
        recon = (p * (torch.log(p + eps) - torch.log(y_target + eps))).sum(1).mean()
        pk = self.prior_kl(mu, logsigma)
        return lam * recon + pk, recon, pk

    def topic_embeddings(self) -> torch.Tensor:
        return self.alphas.weight  # (K, rho_size==d)


@register_topic_model("softltm")
class SoftLTMTopicModel(TopicModel):
    """Soft-label ProdLDA whose ETM decoder yields native retriever-space centroids."""

    name = "softltm"

    def __init__(
        self,
        num_topics: int,
        encoder: "RetrieverEncoder",
        centroid_normalize: bool = True,
        device: str | None = None,
        label_lm: str = "meta-llama/Llama-3.2-1B-Instruct",
        prompt_template: str = DEFAULT_LABEL_PROMPT,
        soft_label_tau: float = 3.0,
        recon_lambda: float = 1e3,
        t_hidden: int = 200,
        dropout: float = 0.2,
        prior_a: float = 0.02,
        dtype: torch.dtype = torch.bfloat16,
        lazy_label_lm: bool = True,
        **kwargs,
    ):
        super().__init__(num_topics, encoder, centroid_normalize, device)
        self.label_lm_name = label_lm
        self.prompt_template = prompt_template
        self.soft_label_tau = float(soft_label_tau)
        self.recon_lambda = float(recon_lambda)
        self.t_hidden = int(t_hidden)
        self.dropout = float(dropout)
        self.prior_a = float(prior_a)
        self.lm_dtype = dtype
        self.lazy_label_lm = bool(lazy_label_lm)

        self.vocab: list[str] | None = None
        self._label_lm: _LabelLM | None = None
        self.core: _ProdLDACore | None = None
        self.train_losses: list[float] = []
        self._pass2_lm_calls: int | None = None

    # ---- interface ----
    @property
    def supports_native_centroids(self) -> bool:
        return True

    def _ensure_label_lm(self) -> _LabelLM:
        if self._label_lm is None:
            if self.vocab is None:
                raise RuntimeError("vocab must be set before building the label LM.")
            self._label_lm = _LabelLM(
                self.label_lm_name,
                self.vocab,
                self.prompt_template,
                device=self.device,
                dtype=self.lm_dtype,
            )
        return self._label_lm

    @torch.no_grad()
    def _embed_vocab(self, vocab: list[str], batch_size: int = 256) -> torch.Tensor:
        rho = self.encoder.encode(vocab, is_query=False, batch_size=batch_size)
        rho = F.normalize(rho.float(), p=2, dim=-1)
        return rho.to(self.device)

    @torch.no_grad()
    def _compute_targets(
        self, texts: list[str], batch_size: int = 8
    ) -> tuple[torch.Tensor, torch.Tensor]:
        label_lm = self._ensure_label_lm()
        x_emb, logits_v = label_lm(texts, batch_size=batch_size)
        y_target = F.softmax(logits_v / self.soft_label_tau, dim=-1)
        return x_emb.float(), y_target.float()

    @torch.no_grad()
    def encode_topic_distribution(
        self, texts: list[str], batch_size: int = 8
    ) -> torch.Tensor:
        if self.core is None:
            raise RuntimeError("SoftLTM core not built; call fit(...) first.")
        label_lm = self._ensure_label_lm()
        self.core.eval()
        rows: list[torch.Tensor] = []
        for start in range(0, len(texts), batch_size):
            sub = texts[start : start + batch_size]
            x_emb, _ = label_lm(sub, batch_size=batch_size)
            theta, _ = self.core.get_theta(x_emb.to(self.device))
            rows.append(theta.float().cpu())
        if not rows:
            return torch.empty(0, self.num_topics, dtype=torch.float32)
        return torch.cat(rows, 0)

    def _native_centroids(self) -> torch.Tensor:
        if self.core is None:
            raise RuntimeError("SoftLTM core not built; call fit(...) first.")
        return self.core.topic_embeddings()  # (K, d), un-normalized

    def fit(self, corpus: "TopicCorpus", cfg: "TopicTrainConfig") -> None:
        vocab = corpus.vocab or corpus.build_vocab(cfg.vocab_size, cfg.min_word_freq)
        self.vocab = list(vocab)
        label_lm = self._ensure_label_lm()
        rho = self._embed_vocab(self.vocab)
        self.core = _ProdLDACore(
            num_topics=self.num_topics,
            vocab_size=len(self.vocab),
            lm_dim=label_lm.hidden_dim,
            rho_size=self.embedding_dim,
            rho_init=rho,
            t_hidden=self.t_hidden,
            dropout=self.dropout,
            prior_a=self.prior_a,
        ).to(self.device)
        self._init_topic_embeddings = self.core.topic_embeddings().detach().clone()

        docs = list(corpus.documents)

        # PASS 1 (no grad): compute + cache the fixed (x_emb, y_target) targets to disk.
        calls_before = label_lm.call_count
        x_all, y_all = self._compute_targets(docs, batch_size=cfg.batch_size)
        cache_dir = tempfile.mkdtemp(prefix="softltm_cache_")
        cache_path = os.path.join(cache_dir, "targets.safetensors")
        from safetensors.torch import load_file, save_file

        save_file({"x_emb": x_all.contiguous(), "y_target": y_all.contiguous()}, cache_path)
        cached = load_file(cache_path)  # memory-mapped reuse
        x_cached, y_cached = cached["x_emb"], cached["y_target"]
        calls_after_pass1 = label_lm.call_count

        # PASS 2: train ONLY the core on the cached targets (label LM is never called).
        optimizer = torch.optim.Adam(self.core.parameters(), lr=cfg.lr)
        rng = random.Random(cfg.seed)
        torch.manual_seed(cfg.seed)
        N = x_cached.shape[0]
        self.train_losses = []
        self.core.train()
        for epoch in range(cfg.epochs):
            order = list(range(N))
            rng.shuffle(order)
            ep_loss = ep_recon = ep_pk = 0.0
            n_b = 0
            for start in range(0, N, cfg.batch_size):
                idxs = order[start : start + cfg.batch_size]
                if len(idxs) < 2:  # BatchNorm needs >=2 samples in training
                    continue
                xb = x_cached[idxs].to(self.device)
                yb = y_cached[idxs].to(self.device)
                loss, recon, pk = self.core(xb, yb, lam=self.recon_lambda)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                ep_loss += float(loss.detach())
                ep_recon += float(recon.detach())
                ep_pk += float(pk.detach())
                n_b += 1
            denom = max(n_b, 1)
            avg = ep_loss / denom
            self.train_losses.append(avg)
            print(
                f"[softltm] epoch {epoch}: loss={avg:.4f} "
                f"recon={ep_recon / denom:.4f} prior_kl={ep_pk / denom:.4f}"
            )

        self._pass2_lm_calls = label_lm.call_count - calls_after_pass1
        self.core.eval()
        self.set_centroid_source(cfg.centroid_source)

    # ---- persistence ----
    def _save_state(self, out_dir: str) -> None:
        if self.core is None:
            raise RuntimeError("Nothing to save; SoftLTM core not built.")
        torch.save(self.core.state_dict(), os.path.join(out_dir, "softltm_core.pt"))
        with open(os.path.join(out_dir, "softltm_config.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "vocab": self.vocab,
                    "label_lm": self.label_lm_name,
                    "prompt_template": self.prompt_template,
                    "soft_label_tau": self.soft_label_tau,
                    "recon_lambda": self.recon_lambda,
                    "t_hidden": self.t_hidden,
                    "dropout": self.dropout,
                    "prior_a": self.prior_a,
                },
                f,
            )

    def _load_state(self, in_dir: str) -> None:
        with open(os.path.join(in_dir, "softltm_config.json"), "r", encoding="utf-8") as f:
            cfg = json.load(f)
        self.vocab = cfg["vocab"]
        self.label_lm_name = cfg["label_lm"]
        self.prompt_template = cfg["prompt_template"]
        self.soft_label_tau = float(cfg["soft_label_tau"])
        self.recon_lambda = float(cfg["recon_lambda"])
        self.t_hidden = int(cfg["t_hidden"])
        self.dropout = float(cfg["dropout"])
        self.prior_a = float(cfg["prior_a"])

        # Rebuild rho via the retriever encoder, then the core; defer the label LM.
        rho = self._embed_vocab(self.vocab)
        # lm_dim must match the saved core; infer from the saved encoder.0 weight.
        sd = torch.load(
            os.path.join(in_dir, "softltm_core.pt"), map_location="cpu", weights_only=False
        )
        lm_dim = sd["encoder.0.weight"].shape[1]
        self.core = _ProdLDACore(
            num_topics=self.num_topics,
            vocab_size=len(self.vocab),
            lm_dim=lm_dim,
            rho_size=self.embedding_dim,
            rho_init=rho,
            t_hidden=self.t_hidden,
            dropout=self.dropout,
            prior_a=self.prior_a,
        ).to(self.device)
        self.core.load_state_dict(sd, strict=True)
        self.core.to(self.device)
        self.core.eval()
        self._label_lm = None  # rebuilt lazily on first encode
