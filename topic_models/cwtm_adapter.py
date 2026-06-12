"""CWTM adapter — wraps the cloned CWTM (Fang et al., LREC-COLING 2024).

CWTM is a prefix-tuned ``BertForMaskedLM`` that produces gated per-token
``word_topics`` summed into a normalized ``doc_topics`` simplex (exposed via
``transform()['document_topic_distributions']``). It is architecturally tied to an
MLM backbone and has **no embedding-space topic vectors**, so per the accepted
design caveat this adapter:

* keeps a configurable MLM backbone (default ``bert-base-uncased``) — it cannot use
  the retriever's decoder embedding model (Qwen3-Embedding-4B) as a backbone,
* takes ``K`` and trains CWTM on the WikiWeb2M corpus (``corpus.vocab`` is unused —
  CWTM relies on its own BERT subword vocabulary),
* reports ``supports_native_centroids == False`` so the base class maps topics into
  the **retriever** space via empirical centroids (responsibility-weighted mean of
  retriever chunk embeddings). The retriever LM defines the centroid space; it is
  simply not CWTM's encoder.

Environment-compat shims (the repo ships ``transformers==5.9``; CWTM was written for
4.x). None of these change CWTM's topic math — they only adapt deprecated/changed
transformers APIs so the *same* computation runs:

1. ``transformers.AdamW`` was removed in v5 → a thin ``torch.optim.AdamW`` subclass
   that accepts and ignores ``correct_bias``.
2. ``PrefixEncoder`` hardcodes bert-base dims (768/12); it is rebuilt to match the
   actual backbone config so non-base backbones work.
3. v5 BERT requires ``past_key_values`` as a ``Cache`` → the legacy prefix tuple is
   converted to a ``DynamicCache`` at the BERT-submodule boundary.
4. ``get_extended_attention_mask``'s 3rd positional is now ``dtype`` (was device).
5. v5 ``BertLayer.forward`` returns a tensor (was a tuple) → ``tran1``/``tran2`` are
   wrapped to restore the tuple output CWTM indexes with ``[0]``.
"""

from __future__ import annotations

import json
import os
import sys
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from topic_models.base import TopicModel, TopicTrainConfig
from topic_models.registry import register_topic_model

if TYPE_CHECKING:
    from src.encoders.retriever_encoder import RetrieverEncoder
    from topic_models.wikiweb2m import TopicCorpus


_CWTM_CACHE: dict = {}


def _import_cwtm(repo_path: str):
    """Import the cloned CWTM module, installing the transformers-v5 AdamW shim."""
    if "module" in _CWTM_CACHE:
        return _CWTM_CACHE["module"], _CWTM_CACHE["PrefixEncoder"]

    import transformers

    if not hasattr(transformers, "AdamW"):
        class _AdamWShim(torch.optim.AdamW):  # noqa: D401 - compat shim
            def __init__(self, *args, correct_bias=None, **kwargs):
                super().__init__(*args, **kwargs)

        transformers.AdamW = _AdamWShim

    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)
    import importlib

    model_mod = importlib.import_module("model")
    _CWTM_CACHE["module"] = model_mod
    _CWTM_CACHE["PrefixEncoder"] = model_mod.PrefixEncoder
    return model_mod, model_mod.PrefixEncoder


class _LayerTupleWrap(nn.Module):
    """Restore the tuple output that v5 ``BertLayer`` no longer returns."""

    def __init__(self, layer: nn.Module):
        super().__init__()
        self.layer = layer

    def forward(self, *args, **kwargs):
        out = self.layer(*args, **kwargs)
        return out if isinstance(out, tuple) else (out,)


@register_topic_model("cwtm")
class CWTMTopicModel(TopicModel):
    """CWTM wrapper. Topics are mapped to the retriever space via empirical centroids."""

    name = "cwtm"

    def __init__(
        self,
        num_topics: int,
        encoder: "RetrieverEncoder",
        centroid_normalize: bool = True,
        device: str | None = None,
        backbone: str = "bert-base-uncased",
        pre_seq_len: int = 10,
        dropout_rate: float = 0.2,
        doc_topic_prior: float = 0.1,
        topic_word_prior: float = 0.1,
        cwtm_repo_path: str = "third_party/CWTM",
        **kwargs,
    ):
        super().__init__(num_topics, encoder, centroid_normalize, device)
        self.backbone_name = backbone
        self.pre_seq_len = int(pre_seq_len)
        self.dropout_rate = float(dropout_rate)
        self.doc_topic_prior = float(doc_topic_prior)
        self.topic_word_prior = float(topic_word_prior)
        self.cwtm_repo_path = cwtm_repo_path

        # CWTM construction downloads the MLM backbone; defer it until fit/load so a
        # registry-driven load() rebuilds with the saved backbone (not the default).
        self.cwtm = None
        self._requested_centroid_source = "empirical"  # forced; no native topics

    # ---- interface ----
    @property
    def supports_native_centroids(self) -> bool:
        return False

    def _native_centroids(self) -> torch.Tensor | None:
        return None

    def _ensure_built(self) -> None:
        if self.cwtm is not None:
            return
        model_mod, PrefixEncoder = _import_cwtm(self.cwtm_repo_path)
        cwtm = model_mod.CWTM(
            num_topics=self.num_topics,
            backbone=self.backbone_name,
            pre_seq_len=self.pre_seq_len,
            dropout_rate=self.dropout_rate,
            doc_topic_prior=self.doc_topic_prior,
            topic_word_prior=self.topic_word_prior,
            device=self.device,
        )
        cwtm.to(self.device)
        assert cwtm.latent_size == self.num_topics, (
            f"CWTM latent_size {cwtm.latent_size} != num_topics {self.num_topics}"
        )
        self._install_compat(cwtm, PrefixEncoder)
        self.cwtm = cwtm

    def _install_compat(self, cwtm, PrefixEncoder) -> None:
        """Apply the transformers-v5 compatibility shims (see module docstring)."""
        from transformers.cache_utils import DynamicCache

        cfg = cwtm.backbone.config
        # (2) Rebuild PrefixEncoder to match the actual backbone dims.
        cwtm.prefix_encoder = PrefixEncoder(
            hidden_size=cfg.hidden_size,
            prefix_hidden_size=512,
            pre_seq_len=cwtm.pre_seq_len,
            num_hidden_layers=cfg.num_hidden_layers,
            prefix_projection=True,
        ).to(self.device)

        # (5) Restore tuple output from the copied BertLayers.
        cwtm.tran1 = _LayerTupleWrap(cwtm.tran1).to(self.device)
        cwtm.tran2 = _LayerTupleWrap(cwtm.tran2).to(self.device)

        # (3) Convert the legacy prefix tuple to a DynamicCache for v5 BERT.
        bert = cwtm.backbone.bert
        orig_forward = bert.forward

        def _bert_forward(*args, **kwargs):
            pkv = kwargs.get("past_key_values", None)
            if isinstance(pkv, (tuple, list)):
                cache = DynamicCache()
                for i, c in enumerate(pkv):
                    cache.update(c[0], c[1], i)
                kwargs["past_key_values"] = cache
            return orig_forward(*args, **kwargs)

        bert.forward = _bert_forward

        # (4) get_extended_attention_mask: 3rd positional is now dtype, not device.
        orig_geam = cwtm.backbone.get_extended_attention_mask

        def _geam(attention_mask, input_shape, dtype=None, **kw):
            if not isinstance(dtype, torch.dtype):
                dtype = torch.float32
            return orig_geam(attention_mask, input_shape, dtype=dtype)

        cwtm.backbone.get_extended_attention_mask = _geam

    @torch.no_grad()
    def encode_topic_distribution(
        self, texts: list[str], batch_size: int = 8
    ) -> torch.Tensor:
        self._ensure_built()
        self.cwtm.eval()
        # CWTM's tokenizer chokes on empty/whitespace-only input; substitute a space.
        safe = [t if (isinstance(t, str) and t.strip()) else " " for t in texts]
        out = self.cwtm.transform(safe, batch_size=batch_size)
        dt = out["document_topic_distributions"]
        theta = torch.as_tensor(dt, dtype=torch.float32).cpu()
        if theta.ndim == 1:
            theta = theta.unsqueeze(0)
        theta = theta.clamp_min(0.0)
        s = theta.sum(dim=1, keepdim=True).clamp_min(1e-12)
        theta = theta / s
        return theta

    def fit(self, corpus: "TopicCorpus", cfg: "TopicTrainConfig") -> None:
        self._ensure_built()
        self.cwtm.train()
        # CWTM ignores corpus.vocab (uses its own BERT subword vocab internally).
        self.cwtm.fit(
            corpus.documents, iterations=cfg.epochs, batch_size=cfg.batch_size
        )
        self.cwtm.eval()
        # No native topics: centroids are always empirical (handled by the base class).
        self._requested_centroid_source = "empirical"

    # ---- persistence ----
    def _save_state(self, out_dir: str) -> None:
        self._ensure_built()
        torch.save(self.cwtm.state_dict(), os.path.join(out_dir, "cwtm.pt"))
        with open(os.path.join(out_dir, "cwtm_config.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "backbone": self.backbone_name,
                    "num_topics": self.num_topics,
                    "pre_seq_len": self.pre_seq_len,
                    "dropout_rate": self.dropout_rate,
                    "doc_topic_prior": self.doc_topic_prior,
                    "topic_word_prior": self.topic_word_prior,
                },
                f,
            )

    def _load_state(self, in_dir: str) -> None:
        with open(os.path.join(in_dir, "cwtm_config.json"), "r", encoding="utf-8") as f:
            cfg = json.load(f)
        self.backbone_name = cfg["backbone"]
        self.pre_seq_len = int(cfg["pre_seq_len"])
        self.dropout_rate = float(cfg["dropout_rate"])
        self.doc_topic_prior = float(cfg["doc_topic_prior"])
        self.topic_word_prior = float(cfg["topic_word_prior"])
        self.cwtm = None  # force rebuild with the saved backbone
        self._ensure_built()
        sd = torch.load(os.path.join(in_dir, "cwtm.pt"), map_location="cpu", weights_only=False)
        self.cwtm.load_state_dict(sd, strict=True)
        self.cwtm.to(self.device)
        self.cwtm.eval()
