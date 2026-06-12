"""Retriever encoder ``f_psi`` for CompassRAG.

A frozen ``Qwen/Qwen3-Embedding-4B`` encoder with last-token pooling + L2
normalization. It wraps a :class:`~src.encoders.backbone.Qwen3TextBackbone`
so that tokenization and pooling are *byte-identical* to Phase 1. This guarantees
``encode(doc, is_query=False)`` lands in the exact same embedding space as the
CEMTM document embedding ``e_d`` for the same text, and lets a single 4B model be
shared (no second download) via :meth:`from_backbone`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.encoders.backbone import Qwen3TextBackbone

_DEFAULT_QUERY_INSTRUCTION = (
    "Given a search query, retrieve relevant passages that answer it"
)


class RetrieverEncoder(nn.Module):
    """``f_psi``. Frozen Qwen3-Embedding-4B with last-token pooling + L2 normalize."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-4B",
        device: str | None = None,
        dtype: torch.dtype = torch.bfloat16,
        max_length: int = 2048,
        query_instruction: str = _DEFAULT_QUERY_INSTRUCTION,
    ):
        super().__init__()
        self.query_instruction = query_instruction
        self.backbone = Qwen3TextBackbone(
            model_name, device=device, dtype=dtype, max_length=max_length, freeze=True
        )
        self._post_init()

    def _post_init(self) -> None:
        # Share the underlying HF model + tokenizer from the backbone.
        self.model = self.backbone.model
        self.tokenizer = self.backbone.tokenizer
        self.device_ = self.backbone.device_
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

    @classmethod
    def from_backbone(
        cls,
        backbone: "Qwen3TextBackbone",
        query_instruction: str | None = None,
    ) -> "RetrieverEncoder":
        """Build a retriever that shares ``backbone.model`` + ``backbone.tokenizer``.

        No second model download/initialization occurs: the returned encoder reuses
        the *same* Python objects so ``e_c`` lands in the same space as the CEMTM
        document embedding.
        """
        self = cls.__new__(cls)
        nn.Module.__init__(self)
        self.query_instruction = (
            query_instruction if query_instruction is not None else _DEFAULT_QUERY_INSTRUCTION
        )
        self.backbone = backbone
        self._post_init()
        return self

    @property
    def dim(self) -> int:
        return int(self.model.config.hidden_size)

    @torch.no_grad()
    def encode(
        self, texts: list[str], batch_size: int = 16, is_query: bool = False
    ) -> torch.Tensor:
        """Encode texts into ``(B, d)`` float32, L2-normalized embeddings on CPU.

        When ``is_query`` is True each text is wrapped with the Qwen3-Embedding
        query convention ``"Instruct: {instruction}\\nQuery: {text}"``. Documents
        and chunks are encoded with no instruction prefix. Tokenization (left-pad,
        EOS append, truncation) and pooling are delegated to the shared backbone,
        so document embeddings match the CEMTM ``e_d`` exactly.
        """
        if is_query:
            prepared = [
                f"Instruct: {self.query_instruction}\nQuery: {t}" for t in texts
            ]
        else:
            prepared = list(texts)

        out_chunks: list[torch.Tensor] = []
        for start in range(0, len(prepared), batch_size):
            sub = prepared[start : start + batch_size]
            inputs = self.backbone._tokenize(sub)
            attn = inputs["attention_mask"]

            outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)
            last_hidden = outputs.hidden_states[-1]  # (B, L, d)

            pooled = Qwen3TextBackbone.pool_last_token(last_hidden, attn)  # (B, d)
            pooled = F.normalize(pooled.float(), p=2, dim=-1)
            out_chunks.append(pooled.cpu())

        if not out_chunks:
            return torch.empty(0, self.dim, dtype=torch.float32)
        return torch.cat(out_chunks, dim=0)
