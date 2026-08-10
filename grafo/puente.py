#!/usr/bin/env python3
"""
Deja importable el RAG de la carpeta de arriba sin copiar una línea.

Este proyecto no reimplementa la recuperación: la envuelve. `rag.py`, `store.py`
y `chunking.py` se importan del RAG original y el índice se lee de ahí mismo
(solo lectura). El entorno virtual, en cambio, es otro a propósito: el del RAG
sirve el proceso web publicado y no se toca.

Importar este módulo tiene que ir ANTES que cualquier `from rag import ...`,
porque store.py lee la ruta del índice del entorno en tiempo de importación.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# El RAG es el directorio padre de este: `grafo/` vive dentro del repo del RAG.
# RAG_DIR permite apuntar a otra copia (p. ej. un índice montado en otro sitio).
RAG_DIR = Path(os.getenv("RAG_DIR", Path(__file__).resolve().parent.parent)).expanduser()

if not (RAG_DIR / "rag.py").exists():
    raise SystemExit(
        f"No encuentro el RAG en {RAG_DIR}. Indica su ruta con RAG_DIR=/ruta/al/rag"
    )

# Misma configuración con la que corre la demo publicada (deploy/rag-demo.service),
# para que los números de este grafo se puedan comparar con los de evaluate.py.
os.environ.setdefault("RAG_INDEX_DIR", str(RAG_DIR / "index"))
os.environ.setdefault("RAG_MODEL", "e5")
os.environ.setdefault("RAG_MODE", "semantic")
os.environ.setdefault("RAG_TOP_K", "3")
os.environ.setdefault("RAG_MAX_CHARS", "2500")

if str(RAG_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_DIR))
