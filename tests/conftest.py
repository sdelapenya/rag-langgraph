"""Piezas compartidas por los tests.

El índice de las pruebas se construye a mano (`Index(...)`) en vez de con
`Index.build(...)`: build baja el modelo de embeddings y tarda segundos, y
nada de lo que se prueba aquí depende de que los vectores signifiquen algo.
Con vectores ortogonales inventados los tests corren en milisegundos y sin red.
"""
from __future__ import annotations

import numpy as np
import pytest

from chunking import Chunk
from store import Index


def chunk(n: int, section: str, page: int, text: str, *, doc: str = "et",
          part: int = 1, n_parts: int = 1) -> Chunk:
    return Chunk(
        id=f"{doc}:{n:04d}",
        doc_id=doc,
        doc_title="Estatuto",
        section=section,
        page=page,
        part=part,
        n_parts=n_parts,
        text=text,
    )


@pytest.fixture
def indice() -> Index:
    """Dos partes del artículo 34 (páginas 10 y 11) y una del 35 (página 12)."""
    chunks = [
        chunk(0, "Artículo 34", 10, "Artículo 34\nLa jornada ordinaria.",
              part=1, n_parts=2),
        chunk(1, "Artículo 34", 11, "Artículo 34\nDescanso entre jornadas.",
              part=2, n_parts=2),
        chunk(2, "Artículo 35", 12, "Artículo 35\nHoras extraordinarias."),
    ]
    return Index(chunks, np.eye(3, dtype=np.float32), "minilm")
