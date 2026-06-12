"""Text-embedding backbone for CompassRAG.

This module wraps a Hugging Face text encoder (default ``Qwen/Qwen3-Embedding-4B``)
and matches the call contract that CEMTM's ``vlm2vec`` module exposes: given a
batch of texts it returns per-token last-layer hidden states ``H`` and a pooled,
L2-normalized document embedding ``e_d``.

The pooling + normalization convention defined here (last non-pad token pooling)
is the *single source of truth* for the retriever embedding space and is reused by
Phase 2 via :meth:`Qwen3TextBackbone.pool_last_token`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


def _pick_device(device: str | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Qwen3TextBackbone(nn.Module):
    """Frozen text backbone producing token hidden states and a pooled embedding.

    Args:
        model_name: HF model id loaded with ``AutoModel``.
        device: Torch device string; ``None`` auto-selects cuda/mps/cpu.
        dtype: Compute dtype for the backbone (default ``bfloat16``).
        max_length: Maximum tokenized length (including the appended EOS token).
        freeze: If ``True``, parameters are frozen and the module is set to eval.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-4B",
        device: str | None = None,
        dtype: torch.dtype = torch.bfloat16,
        max_length: int = 2048,
        freeze: bool = True,
    ):
        super().__init__()
        self.model_name = model_name
        self.max_length = int(max_length)
        self.dtype = dtype
        self.device_ = _pick_device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Last-token pooling requires left padding so the final position (-1) is
        # always a valid (non-pad) token for every row in the batch.
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            # Fall back to EOS (or any available special token) as the pad token.
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            elif self.tokenizer.sep_token is not None:
                self.tokenizer.pad_token = self.tokenizer.sep_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        # The token appended to every sequence to act as the pooling position.
        # Qwen3-Embedding uses EOS; BERT-style models expose SEP instead.
        if self.tokenizer.eos_token_id is not None:
            self._eos_id = self.tokenizer.eos_token_id
        elif self.tokenizer.sep_token_id is not None:
            self._eos_id = self.tokenizer.sep_token_id
        else:
            self._eos_id = None

        self.model = AutoModel.from_pretrained(model_name, dtype=dtype)
        if self.tokenizer.pad_token == "[PAD]" and self.model.get_input_embeddings().num_embeddings < len(self.tokenizer):
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.to(self.device_)

        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()

    @property
    def hidden_dim(self) -> int:
        return int(self.model.config.hidden_size)

    @staticmethod
    def pool_last_token(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Pool the last *non-pad* token hidden state of each sequence.

        This is the canonical pooling used to build the retriever embedding space.
        It is robust to both left and right padding:

        * With left padding (the convention used here) the last token is index ``-1``.
        * With right padding the last token is ``attention_mask.sum(dim=1) - 1``.

        Args:
            last_hidden: ``(B, L, D)`` last-layer hidden states.
            attention_mask: ``(B, L)`` 1 for valid tokens, 0 for padding.

        Returns:
            ``(B, D)`` pooled embedding (not normalized).
        """
        # Detect left padding: every row has a valid token in the final column.
        left_padding = bool((attention_mask[:, -1].sum() == attention_mask.shape[0]).item())
        if left_padding:
            return last_hidden[:, -1]
        seq_lengths = attention_mask.sum(dim=1) - 1  # (B,)
        batch_idx = torch.arange(last_hidden.shape[0], device=last_hidden.device)
        return last_hidden[batch_idx, seq_lengths]

    def _tokenize(self, texts: list[str]) -> dict[str, torch.Tensor]:
        # Encode empty / whitespace-only strings as a single space so the model
        # always receives at least one real token.
        cleaned = [t if (isinstance(t, str) and t.strip() != "") else " " for t in texts]

        # Reserve room for the appended EOS token, then append it per sequence.
        reserve = 1 if self._eos_id is not None else 0
        encoded = self.tokenizer(
            cleaned,
            padding=False,
            truncation=True,
            max_length=self.max_length - reserve if reserve else self.max_length,
            add_special_tokens=True,
        )
        if self._eos_id is not None:
            for i, ids in enumerate(encoded["input_ids"]):
                ids.append(self._eos_id)
                # Keep all parallel fields aligned with the extended input_ids.
                if "attention_mask" in encoded:
                    encoded["attention_mask"][i].append(1)
                if "token_type_ids" in encoded:
                    encoded["token_type_ids"][i].append(0)

        batch = self.tokenizer.pad(encoded, padding=True, return_tensors="pt")
        return {k: v.to(self.device_) for k, v in batch.items()}

    @torch.no_grad()
    def encode_tokens(
        self, texts: list[str], batch_size: int = 8
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Encode texts into per-token hidden states and a pooled doc embedding.

        Returns:
            ``(H_list, e_d)`` where

            * ``H_list`` is a list of length ``B``; each element is an
              ``(N_b, D)`` ``float32`` tensor **on CPU** holding the last-layer
              hidden states of the *valid* (non-pad) tokens of that text. They are
              returned on CPU in float32 to bound GPU memory for long batches.
            * ``e_d`` is ``(B, D)`` ``float32`` on the backbone device: the
              L2-normalized last-token (EOS) pooled embedding.
        """
        h_list: list[torch.Tensor] = []
        pooled_chunks: list[torch.Tensor] = []

        for start in range(0, len(texts), batch_size):
            sub = texts[start : start + batch_size]
            inputs = self._tokenize(sub)
            attn = inputs["attention_mask"]

            outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)
            last_hidden = outputs.hidden_states[-1]  # (B, L, D)

            pooled = self.pool_last_token(last_hidden, attn)  # (B, D)
            pooled = F.normalize(pooled.float(), p=2, dim=-1)
            pooled_chunks.append(pooled)

            mask_bool = attn.bool()
            for b in range(last_hidden.shape[0]):
                valid = last_hidden[b][mask_bool[b]]  # (N_b, D)
                h_list.append(valid.float().cpu())

        e_d = torch.cat(pooled_chunks, dim=0) if pooled_chunks else torch.empty(
            0, self.hidden_dim, device=self.device_
        )
        return h_list, e_d
