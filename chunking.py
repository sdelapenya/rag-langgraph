#!/usr/bin/env python3
"""
Extracción y troceado de PDFs con estructura de artículos (BOE, reglamentos,
manuales numerados).

La diferencia con un troceado ciego cada N caracteres es que aquí cada fragmento
conserva de qué artículo y de qué página sale, y eso es lo que permite luego
citar la fuente exacta en la respuesta.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path

import pdfplumber

MAX_CHARS = int(os.getenv("RAG_CHUNK", "1200"))       # tamaño máximo de un fragmento
OVERLAP_CHARS = int(os.getenv("RAG_OVERLAP", "150"))  # solape entre subfragmentos

# Ruido repetido en cada página de los PDFs del BOE
_NOISE_LINES = {
    "BOLETÍN OFICIAL DEL ESTADO",
    "LEGISLACIÓN CONSOLIDADA",
    "TEXTO CONSOLIDADO",
}
_PAGE_FOOTER = re.compile(r"^Página\s+\d+$")
_TOC_LINE = re.compile(r"(?:\.\s*){5,}")  # puntos de relleno del índice
_BODY_MARKER = "TEXTO CONSOLIDADO"  # en el BOE, aquí acaba el índice y empieza la ley

# Encabezados que abren una sección nueva. Sin IGNORECASE y exigiendo que tras
# el número no venga otro dígito: si no, "lo previsto en el artículo 83.2" se
# tomaría por un encabezado y partiría el documento por la mitad de una frase.
_SECTION_HEAD = re.compile(
    r"^(?:"
    r"Artículo\s+(?:único|\d+\s*(?:bis|ter|quáter|quinquies)?)\.(?!\d)"
    r"|Disposición\s+(?:adicional|transitoria|derogatoria|final)\s+\S+"
    r")"
)


@dataclass
class Chunk:
    """Un fragmento indexable con su procedencia."""

    id: str
    doc_id: str
    doc_title: str
    section: str
    page: int
    part: int
    n_parts: int
    text: str

    def cite(self) -> str:
        """Cita legible para mostrar bajo la respuesta."""
        return f"{self.doc_title} — {self.section} (pág. {self.page})"

    def as_dict(self) -> dict:
        return asdict(self)


def clean_page(text: str) -> str:
    """Quita cabeceras, pies y líneas del índice."""
    kept = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped in _NOISE_LINES or _PAGE_FOOTER.match(stripped):
            continue
        if _TOC_LINE.search(stripped):
            continue
        kept.append(stripped)
    return "\n".join(kept)


def extract_pages(pdf_path: Path) -> list[str]:
    """
    Texto limpio de cada página (posición 0 = página 1), sin el índice inicial.

    El índice del BOE es texto plausible y bien escrito, así que si se indexa
    compite con el articulado real y el modelo acaba citando la página del
    índice en vez del artículo. Se corta en la marca TEXTO CONSOLIDADO.
    """
    with pdfplumber.open(pdf_path) as pdf:
        raw = [page.extract_text() or "" for page in pdf.pages]

    body_start = next((i for i, t in enumerate(raw) if _BODY_MARKER in t), None)
    pages = []
    for i, text in enumerate(raw):
        if body_start is not None:
            if i < body_start:
                pages.append("")
                continue
            if i == body_start:
                text = text.split(_BODY_MARKER, 1)[1]
        pages.append(clean_page(text))
    return pages


def _split_sections(pages: list[str]) -> list[tuple[str, list[tuple[str, int]]]]:
    """
    Recorre el documento línea a línea y lo parte por encabezados de artículo.

    Devuelve tuplas (encabezado, [(línea, página), ...]). La página se guarda
    línea a línea, no por sección: un artículo largo ocupa varias páginas y la
    cita tiene que apuntar a la página del fragmento concreto, no a la primera.
    """
    sections: list[tuple[str, list[tuple[str, int]]]] = []
    current_head = "Preámbulo"
    buffer: list[tuple[str, int]] = []

    pending_head: list[str] | None = None  # encabezado partido en varias líneas

    for page_no, page_text in enumerate(pages, start=1):
        for line in page_text.splitlines():
            # el título del artículo puede venir cortado en dos líneas; se
            # completa mientras no termine en punto y no empiece el articulado
            if pending_head is not None:
                if not re.match(r"^\d+\.|^[a-z]\)", line) and len(pending_head) < 3:
                    pending_head.append(line)
                    current_head = " ".join(pending_head).strip()
                    if line.rstrip().endswith("."):
                        pending_head = None
                    continue
                pending_head = None

            if _SECTION_HEAD.match(line):
                if buffer:
                    sections.append((current_head, buffer))
                current_head = line.strip()
                buffer = []
                pending_head = None if current_head.endswith(".") else [current_head]
            else:
                buffer.append((line, page_no))
    if buffer:
        sections.append((current_head, buffer))

    return sections


def _split_long(text: str) -> list[tuple[str, int]]:
    """
    Parte un texto largo en trozos con solape.

    Devuelve (trozo, posición de inicio) para poder saber después en qué página
    cae cada trozo.
    """
    if len(text) <= MAX_CHARS:
        return [(text, 0)]

    parts, start = [], 0
    while start < len(text):
        end = min(start + MAX_CHARS, len(text))
        if end < len(text):
            # cortar en el último final de frase o salto de línea del tramo
            cut = max(text.rfind(". ", start, end), text.rfind("\n", start, end))
            if cut > start + MAX_CHARS // 2:
                end = cut + 1
        parts.append((text[start:end].strip(), start))
        if end >= len(text):
            break
        start = max(end - OVERLAP_CHARS, start + 1)
    return [(p, off) for p, off in parts if p]


def _page_map(lines: list[tuple[str, int]]) -> tuple[str, list[int]]:
    """Une las líneas y devuelve el texto y la página de cada carácter."""
    text_parts, page_of_char = [], []
    for line, page_no in lines:
        text_parts.append(line)
        page_of_char.extend([page_no] * (len(line) + 1))  # +1 por el salto
    return "\n".join(text_parts), page_of_char


def chunk_pdf(pdf_path: Path, doc_id: str, doc_title: str) -> list[Chunk]:
    """PDF -> lista de fragmentos con procedencia."""
    pages = extract_pages(pdf_path)
    chunks: list[Chunk] = []

    for head, lines in _split_sections(pages):
        head = head.rstrip(".").strip()
        # sin .strip(): las posiciones tienen que cuadrar con page_of_char
        body, page_of_char = _page_map(lines)
        if len(body) < 40:  # secciones vacías o solo con el título
            continue
        # el encabezado se repite en cada trozo: sin él, un fragmento suelto
        # no dice de qué artículo habla y el modelo pierde el contexto
        pieces = _split_long(body)
        for i, (piece, offset) in enumerate(pieces, start=1):
            page = page_of_char[min(offset, len(page_of_char) - 1)] if page_of_char else 0
            chunks.append(
                Chunk(
                    id=f"{doc_id}:{len(chunks):04d}",
                    doc_id=doc_id,
                    doc_title=doc_title,
                    section=head,
                    page=page,
                    part=i,
                    n_parts=len(pieces),
                    text=f"{head}\n{piece}",
                )
            )
    return chunks
