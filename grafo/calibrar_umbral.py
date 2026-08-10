#!/usr/bin/env python3
"""
De dónde sale el umbral de similitud del nodo `evaluar`.

    python3 calibrar_umbral.py

Recorre el mismo conjunto de evaluación del RAG (24 preguntas: 20 con respuesta
en el corpus y 4 de materias que no regula) y mide la similitud coseno del mejor
fragmento recuperado. Si las dos poblaciones se separan, hay umbral; si se
solapan, no lo hay y el filtro barato no vale — mejor saberlo que suponerlo.

No gasta cuota: las reescrituras se leen de la caché que ya dejó evaluate.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import puente

from grafo import _similitudes  # noqa: E402
from rag import TOP_K  # noqa: E402
from store import DEFAULT_MODE, Index  # noqa: E402

PREGUNTAS = puente.RAG_DIR / "eval-preguntas.json"
REESCRITURAS = puente.RAG_DIR / "eval-reescrituras.json"


def main() -> None:
    preguntas = json.loads(PREGUNTAS.read_text(encoding="utf-8"))["preguntas"]
    cache = json.loads(REESCRITURAS.read_text(encoding="utf-8")) if REESCRITURAS.exists() else {}
    index = Index.load()

    dentro, fuera = [], []
    for p in preguntas:
        extra = [cache[p["id"]]] if cache.get(p["id"]) else None
        hits = index.search(p["pregunta"], k=TOP_K, mode=DEFAULT_MODE,
                            max_per_section=1, extra_queries=extra)
        mejor = max(_similitudes(index, p["pregunta"], hits), default=0.0)
        (fuera if p.get("abstenerse") else dentro).append((mejor, p["id"]))

    for etiqueta, grupo in (("EN el corpus", dentro), ("FUERA del corpus", fuera)):
        print(f"\n  {etiqueta} ({len(grupo)} preguntas)")
        for sim, pid in sorted(grupo):
            print(f"    {sim:.4f}  {pid}")

    minimo_dentro = min(s for s, _ in dentro)
    maximo_fuera = max(s for s, _ in fuera)
    print(f"\n  mínimo dentro  {minimo_dentro:.4f}")
    print(f"  máximo fuera   {maximo_fuera:.4f}")
    if maximo_fuera < minimo_dentro:
        print(f"  -> separan: umbral a mitad de camino = {(maximo_fuera + minimo_dentro) / 2:.2f}")
    else:
        print("  -> se solapan: no hay umbral que separe. El filtro barato solo puede\n"
              "     cortar por debajo del mínimo de dentro, y ahí ya no ahorra nada.")


if __name__ == "__main__":
    main()
