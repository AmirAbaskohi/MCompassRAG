"""Training harness: build + fit a topic backend, then persist it."""

from __future__ import annotations

from typing import TYPE_CHECKING

from topic_models.base import TopicModel, TopicTrainConfig
from topic_models.registry import build_topic_model
from topic_models.wikiweb2m import TopicCorpus

if TYPE_CHECKING:
    from src.encoders.retriever_encoder import RetrieverEncoder

_BOW_BACKENDS = {"etm", "cwtm", "softltm"}


def train_topic_model(
    name: str,
    encoder: "RetrieverEncoder",
    corpus: TopicCorpus,
    cfg: TopicTrainConfig,
    out_dir: str,
    **backend_kwargs,
) -> TopicModel:
    """Build, fit, ground (empirical if needed), and save a topic model."""
    model = build_topic_model(
        name,
        cfg.num_topics,
        encoder,
        centroid_normalize=cfg.centroid_normalize,
        device=cfg.device,
        **backend_kwargs,
    )
    model.set_centroid_source(cfg.centroid_source)

    if name in _BOW_BACKENDS and corpus.vocab is None:
        corpus.build_vocab(cfg.vocab_size, cfg.min_word_freq)

    model.fit(corpus, cfg)
    model.maybe_fit_empirical(corpus, cfg)
    model.save(out_dir)
    return model


def load_topic_model(in_dir: str, encoder: "RetrieverEncoder") -> TopicModel:
    return TopicModel.load(in_dir, encoder)


def build_arg_parser():
    import argparse

    p = argparse.ArgumentParser(
        description="Train a pluggable topic model (cemtm|etm|cwtm|softltm) on WikiWeb2M."
    )
    p.add_argument("--config", required=True, help="Path to a train_topic_model YAML.")
    return p


def main(argv: list[str] | None = None) -> int:
    from src.config import bind, build_encoder, load_yaml, section
    from topic_models.wikiweb2m import load_wikiweb2m

    args = build_arg_parser().parse_args(argv)
    cfg = load_yaml(args.config)

    name = section(cfg, "topic_model", "name", default="cemtm")
    num_topics = int(section(cfg, "topic_model", "num_topics", default=100))

    encoder = build_encoder(cfg)

    train_sec = dict(section(cfg, "train", default={}) or {})
    train_sec["num_topics"] = num_topics
    tcfg = bind(TopicTrainConfig, train_sec)

    corpus_sec = section(cfg, "corpus", default={}) or {}
    corpus = load_wikiweb2m(
        corpus_sec["wikiweb2m_path"],
        max_docs=corpus_sec.get("max_docs"),
        min_doc_chars=corpus_sec.get("min_doc_chars", 200),
    )

    backend_kwargs = section(cfg, "backend_kwargs", default={}) or {}
    out_dir = section(cfg, "output", "save_dir")
    if not out_dir:
        raise ValueError("config is missing output.save_dir")

    train_topic_model(name, encoder, corpus, tcfg, out_dir, **backend_kwargs)
    print(f"[train_topic_model] saved '{name}' (K={num_topics}) to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
