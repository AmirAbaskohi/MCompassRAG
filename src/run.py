"""RAG CLI: build the metadata bank and/or run queries against a served retriever.

Examples::

    python -m src.run --config configs/rag/cemtm.yaml --build-index
    python -m src.run --config configs/rag/cemtm.yaml --query "what is X?"
    python -m src.run --config configs/rag/cemtm.yaml --queries questions.txt
"""

from __future__ import annotations

import argparse
import os


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="CompassRAG: build the index and/or run retrieval queries."
    )
    p.add_argument("--config", required=True, help="Path to a configs/rag/<x>.yaml.")
    p.add_argument(
        "--build-index",
        action="store_true",
        help="Build the metadata bank from the topic model + chunks before serving.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Rebuild the index even if index.out_dir already exists.",
    )
    p.add_argument("--query", default=None, help="A single query to retrieve for.")
    p.add_argument(
        "--queries", default=None, help="File with one query per line (batch mode)."
    )
    p.add_argument(
        "--k", type=int, default=None, help="Override the number of results to return."
    )
    return p


def _print_results(query: str, results) -> None:
    print(f"\nQuery: {query}")
    for r in results:
        text = " ".join(r.text.split())
        if len(text) > 160:
            text = text[:157] + "..."
        print(f"  [{r.rank}] score={r.score:.4f} id={r.chunk_id}  {text}")


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    from src import pipeline
    from src.config import load_yaml, section

    cfg = load_yaml(args.config)

    if args.build_index:
        out_dir = section(cfg, "index", "out_dir")
        if out_dir and os.path.isdir(out_dir) and not args.force:
            print(f"[run] index exists at {out_dir}; skipping (use --force to rebuild).")
        else:
            built = pipeline.build_index_from_config(cfg)
            print(f"[run] built index at {built}")

    queries: list[str] = []
    if args.query:
        queries.append(args.query)
    if args.queries:
        with open(args.queries, "r", encoding="utf-8") as f:
            queries.extend(line.strip() for line in f if line.strip())

    if queries:
        rag = pipeline.serve_from_config(cfg)
        k = args.k if args.k is not None else rag.cfg.k
        if len(queries) == 1:
            _print_results(queries[0], rag.retrieve(queries[0], k=k))
        else:
            for q, res in zip(queries, rag.retrieve_batch(queries, k=k)):
                _print_results(q, res)

    if not args.build_index and not queries:
        print("Nothing to do: pass --build-index and/or --query/--queries.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
