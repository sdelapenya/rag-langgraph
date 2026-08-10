#!/usr/bin/env python3
"""
Construye el índice a partir de los PDFs de corpus/.

    python3 ingest.py                 # modelo por defecto (RAG_MODEL o minilm)
    python3 ingest.py --model e5      # otro modelo, se guarda en su propia carpeta
    python3 ingest.py --corpus otra/  # otro corpus

Cada modelo tiene su índice aparte, así que se pueden comparar sin reindexar.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from chunking import chunk_pdf
from store import DEFAULT_MODEL, MODELS, Index


def load_manifest(corpus_dir: Path) -> list[dict]:
    manifest = corpus_dir / "corpus.json"
    if manifest.exists():
        return json.loads(manifest.read_text(encoding="utf-8"))["documentos"]
    # sin manifiesto: cada PDF es un documento y el título sale del nombre
    return [
        {"archivo": p.name, "doc_id": p.stem, "titulo": p.stem.replace("-", " ").title()}
        for p in sorted(corpus_dir.glob("*.pdf"))
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Indexa los PDFs del corpus")
    parser.add_argument("--corpus", default="corpus", help="carpeta con los PDFs")
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=list(MODELS))
    args = parser.parse_args()

    corpus_dir = Path(args.corpus)
    documentos = load_manifest(corpus_dir)
    if not documentos:
        raise SystemExit(f"No hay PDFs en {corpus_dir}")

    chunks = []
    for doc in documentos:
        path = corpus_dir / doc["archivo"]
        if not path.exists():
            print(f"  ! falta {path}, lo salto")
            continue
        print(f"  Troceando {doc['titulo']}...", end=" ", flush=True)
        doc_chunks = chunk_pdf(path, doc["doc_id"], doc["titulo"])
        chunks.extend(doc_chunks)
        print(f"{len(doc_chunks)} fragmentos")

    print(f"\nGenerando embeddings con {MODELS[args.model]['name']}...")
    started = time.time()
    index = Index.build(chunks, args.model)
    elapsed = time.time() - started

    destino = index.save()
    print(
        f"Índice guardado en {destino}/ — {len(chunks)} fragmentos, "
        f"dim {index.vectors.shape[1]}, {elapsed:.1f} s "
        f"({len(chunks) / elapsed:.0f} fragmentos/s)"
    )


if __name__ == "__main__":
    main()
