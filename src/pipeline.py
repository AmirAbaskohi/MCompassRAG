"""End-to-end pipeline helpers: corpus -> cached index -> served retriever.

``build_index`` is the ONLY component that touches a topic model; serving reads the
cached bank and never needs it. ``topic_models`` is imported lazily inside the
functions below so ``import src`` / ``import src.pipeline`` stays cycle-free.
"""

from __future__ import annotations

import json

from src.config import bind, build_encoder, section
from src.index.chunking import Chunk
from src.index.metadata_bank import IndexConfig, MetadataBank
from src.inference.retrieve import CompassRAG, CompassRAGConfig


def build_index(
    chunks: list[Chunk],
    out_dir: str,
    backbone_name: str = "Qwen/Qwen3-Embedding-4B",
    cemtm_ckpt: str | None = None,
    num_topics: int = 100,
    centroid_source: str = "decoder",
    centroid_normalize: bool = True,
    index_cfg: IndexConfig | None = None,
    device: str | None = None,
    topic_model_name: str = "cemtm",
) -> str:
    """Build and persist the offline metadata bank for ``chunks``.

    Constructs a fresh topic model via the registry (default ``cemtm``); its
    retriever encoder defines the shared embedding space. Returns ``out_dir``.
    """
    from topic_models.registry import build_topic_model  # lazy: avoids import cycle

    topic_model = build_topic_model(
        topic_model_name,
        num_topics,
        encoder=None,
        centroid_normalize=centroid_normalize,
        device=device,
        checkpoint_path=cemtm_ckpt,
        backbone_name=backbone_name,
        centroid_source=centroid_source,
    )
    encoder = topic_model.encoder
    bank = MetadataBank.build(chunks, encoder, topic_model, index_cfg or IndexConfig())
    bank.save(out_dir)
    return out_dir


def serve(bank_dir: str, model_ckpt: str, **kwargs) -> CompassRAG:
    """Reconstruct a served :class:`CompassRAG` from a saved bank + checkpoint."""
    return CompassRAG.from_pretrained(bank_dir, model_ckpt, **kwargs)


# --------------------------------------------------------------------------- #
# Config-driven helpers (used by ``src.run``)                                  #
# --------------------------------------------------------------------------- #
def load_chunks_jsonl(path: str) -> list[Chunk]:
    """Read ``{id,text,doc_id,position}`` records into :class:`Chunk` objects."""
    chunks: list[Chunk] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            chunks.append(
                Chunk(id=str(rec["id"]), text=rec["text"], doc_id=rec.get("doc_id"))
            )
    return chunks


def build_index_from_config(cfg: dict) -> str:
    """Build the metadata bank from a trained topic model + ``index`` section.

    Loads the topic model saved under ``topic_model.dir`` (so centroids are real,
    not random), embeds the chunks at ``index.chunks_path`` with the shared
    encoder, and writes the bank to ``index.out_dir``.
    """
    from topic_models.train_topic_model import load_topic_model  # lazy

    encoder = build_encoder(cfg)

    tm_dir = section(cfg, "topic_model", "dir")
    if not tm_dir:
        raise ValueError("config is missing topic_model.dir")
    topic_model = load_topic_model(tm_dir, encoder)

    idx = section(cfg, "index", default={}) or {}
    out_dir = idx.get("out_dir")
    chunks_path = idx.get("chunks_path")
    if not out_dir or not chunks_path:
        raise ValueError("config is missing index.out_dir or index.chunks_path")

    index_cfg = bind(IndexConfig, idx)
    chunks = load_chunks_jsonl(chunks_path)
    bank = MetadataBank.build(chunks, encoder, topic_model, index_cfg)
    bank.save(out_dir)
    return out_dir


def serve_from_config(cfg: dict) -> CompassRAG:
    """Reconstruct a served :class:`CompassRAG` from a RAG config dict."""
    encoder = build_encoder(cfg)
    bank_dir = section(cfg, "index", "out_dir")
    model_ckpt = section(cfg, "retriever", "model_ckpt")
    if not bank_dir or not model_ckpt:
        raise ValueError("config is missing index.out_dir or retriever.model_ckpt")

    serve = section(cfg, "serve", default={}) or {}
    rag_cfg = bind(CompassRAGConfig, serve)
    return CompassRAG.from_pretrained(
        bank_dir,
        model_ckpt,
        encoder=encoder,
        device=serve.get("device"),
        cfg=rag_cfg,
    )
