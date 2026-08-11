#!/usr/bin/env python3
"""
Índice de búsqueda: embeddings (semántica) + BM25 (palabras) fusionados con RRF.

Ninguno de los dos sirve solo. Los embeddings encuentran «cuánto me pagan si me
echan» aunque el texto diga «indemnización por despido improcedente», pero
fallan con referencias literales tipo «artículo 34.8» o siglas. BM25 hace lo
contrario. La fusión RRF (Reciprocal Rank Fusion) se queda con lo bueno de cada
uno sin tener que calibrar pesos.
"""
from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np

from chunking import OVERLAP_CHARS, Chunk

# Modelos de embeddings disponibles (ONNX en CPU, sin GPU ni coste por llamada).
# e5 exige prefijos distintos para consulta y documento: es como fue entrenado,
# y sin ellos pierde precisión.
MODELS = {
    "minilm": {
        "name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "query_prefix": "",
        "passage_prefix": "",
        "dim": 384,
    },
    "mpnet": {
        "name": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        "query_prefix": "",
        "passage_prefix": "",
        "dim": 768,
    },
    "e5": {
        "name": "intfloat/multilingual-e5-large",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
        "dim": 1024,
    },
}
DEFAULT_MODEL = os.getenv("RAG_MODEL", "minilm")
# semántica pura: el híbrido con BM25 mide peor en este corpus (ver README)
DEFAULT_MODE = os.getenv("RAG_MODE", "semantic")
INDEX_DIR = Path(os.getenv("RAG_INDEX_DIR", "index"))
# fastembed cachea los modelos en /tmp por defecto: eso obliga a volver a
# descargar ~1 GB en cada reinicio y, bajo systemd con PrivateTmp, ni siquiera
# se reutiliza entre arranques.
MODEL_CACHE = Path(os.getenv("RAG_CACHE_DIR", Path.home() / ".cache" / "fastembed"))
BATCH_SIZE = int(os.getenv("RAG_BATCH", "16"))

# Pesos de la fusión híbrida, elegidos con los números de evaluate.py
SEMANTIC_WEIGHT = float(os.getenv("RAG_W_SEMANTIC", "1.0"))
KEYWORD_WEIGHT = float(os.getenv("RAG_W_KEYWORD", "0.4"))
EXPANSION_WEIGHT = float(os.getenv("RAG_W_EXPANSION", "0.7"))

_STOPWORDS = {
    "a", "al", "algo", "algún", "alguna", "algunas", "alguno", "algunos", "ante",
    "antes", "como", "con", "contra", "cual", "cuando", "de", "del", "desde",
    "donde", "durante", "e", "el", "ella", "ellas", "ello", "ellos", "en",
    "entre", "era", "eran", "es", "esa", "esas", "ese", "eso", "esos", "esta",
    "estas", "este", "esto", "estos", "ha", "han", "hasta", "hay", "la", "las",
    "le", "les", "lo", "los", "más", "me", "mi", "mis", "mucho", "muy", "no",
    "nos", "o", "para", "pero", "poco", "por", "porque", "que", "qué", "quien",
    "se", "sea", "según", "ser", "si", "sí", "sin", "sobre", "son", "su", "sus",
    "también", "tanto", "te", "tiene", "tienen", "todo", "todos", "tu", "tus",
    "un", "una", "uno", "unos", "y", "ya",
}


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"\w+", text.lower()) if t not in _STOPWORDS]


class BM25:
    """BM25 Okapi en unas pocas líneas — no hace falta traerse una dependencia."""

    K1 = 1.5
    B = 0.75

    def __init__(self, corpus_tokens: list[list[str]]):
        self.docs = corpus_tokens
        self.n = len(corpus_tokens)
        self.lengths = np.array([len(d) for d in corpus_tokens], dtype=np.float32)
        self.avg_len = float(self.lengths.mean()) if self.n else 0.0
        self.freqs = [Counter(d) for d in corpus_tokens]
        df = Counter()
        for d in corpus_tokens:
            df.update(set(d))
        self.idf = {
            term: math.log(1 + (self.n - n_q + 0.5) / (n_q + 0.5))
            for term, n_q in df.items()
        }

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(self.n, dtype=np.float32)
        for term in tokenize(query):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, freq in enumerate(self.freqs):
                f = freq.get(term, 0)
                if not f:
                    continue
                denom = f + self.K1 * (1 - self.B + self.B * self.lengths[i] / self.avg_len)
                out[i] += idf * (f * (self.K1 + 1)) / denom
        return out


def _rrf(rankings: list[list[int]], weights: list[float] | None = None,
         k: int = 60) -> dict[int, float]:
    """
    Reciprocal Rank Fusion: suma ponderada de 1/(k+posición) de cada lista.

    Con pesos iguales, una lista mala arrastra a la buena. Aquí la semántica
    acierta bastante más que BM25 (medido en evaluate.py), así que pesa más.
    """
    weights = weights or [1.0] * len(rankings)
    fused: dict[int, float] = {}
    # strict: si los pesos no cuadran con las listas, zip descartaría en
    # silencio la última lista y la fusión saldría mal sin avisar
    for ranking, weight in zip(rankings, weights, strict=True):
        for pos, idx in enumerate(ranking, start=1):
            fused[idx] = fused.get(idx, 0.0) + weight / (k + pos)
    return fused


class Index:
    """Fragmentos + sus vectores + BM25, con guardado en disco."""

    def __init__(self, chunks: list[Chunk], vectors: np.ndarray, model_key: str):
        self.chunks = chunks
        self.vectors = vectors
        self.model_key = model_key
        self.bm25 = BM25([tokenize(c.text) for c in chunks])
        self._embedder = None

    # --- construcción y persistencia -------------------------------------

    @staticmethod
    def _embedder_for(model_key: str):
        from fastembed import TextEmbedding  # import perezoso: tarda ~2 s

        MODEL_CACHE.mkdir(parents=True, exist_ok=True)
        return TextEmbedding(
            model_name=MODELS[model_key]["name"], cache_dir=str(MODEL_CACHE)
        )

    @classmethod
    def build(cls, chunks: list[Chunk], model_key: str = DEFAULT_MODEL) -> Index:
        cfg = MODELS[model_key]
        embedder = cls._embedder_for(model_key)
        texts = [cfg["passage_prefix"] + c.text for c in chunks]
        # lotes pequeños y sin multiproceso: con los valores por defecto, un
        # modelo grande llegó a ocupar 8,5 GB de RAM en este servidor
        vectors = np.array(
            list(embedder.embed(texts, batch_size=BATCH_SIZE, parallel=1)),
            dtype=np.float32,
        )
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        index = cls(chunks, vectors, model_key)
        index._embedder = embedder
        return index

    def save(self, directory: Path | None = None) -> Path:
        directory = Path(directory or INDEX_DIR) / self.model_key
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", self.vectors)
        with (directory / "chunks.jsonl").open("w", encoding="utf-8") as fh:
            for chunk in self.chunks:
                fh.write(json.dumps(chunk.as_dict(), ensure_ascii=False) + "\n")
        (directory / "meta.json").write_text(
            json.dumps(
                {
                    "model_key": self.model_key,
                    "model_name": MODELS[self.model_key]["name"],
                    "n_chunks": len(self.chunks),
                    "dim": int(self.vectors.shape[1]),
                    "documents": sorted({c.doc_title for c in self.chunks}),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return directory

    @classmethod
    def load(cls, model_key: str = DEFAULT_MODEL, directory: Path | None = None) -> Index:
        directory = Path(directory or INDEX_DIR) / model_key
        if not (directory / "vectors.npy").exists():
            raise FileNotFoundError(
                f"No hay índice en {directory}. Créalo con: python3 ingest.py"
            )
        vectors = np.load(directory / "vectors.npy")
        chunks = [
            Chunk(**json.loads(line))
            for line in (directory / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return cls(chunks, vectors, model_key)

    @property
    def meta(self) -> dict:
        return {
            "model": MODELS[self.model_key]["name"],
            "n_chunks": len(self.chunks),
            "documents": sorted({c.doc_title for c in self.chunks}),
        }

    # --- contexto ---------------------------------------------------------

    def section_chunks(self, chunk: Chunk) -> list[Chunk]:
        """Todos los trozos del mismo artículo, en orden."""
        return sorted(
            (c for c in self.chunks
             if c.doc_id == chunk.doc_id and c.section == chunk.section),
            key=lambda c: c.part,
        )

    def section_text(self, chunk: Chunk, max_chars: int = 4000) -> tuple[str, str]:
        """
        Devuelve el artículo entero y su cita con el rango de páginas.

        Buscar por fragmentos y responder con fragmentos son dos cosas
        distintas: el trozo que mejor casa con la pregunta ("horas
        extraordinarias") no tiene por qué ser el que lleva el dato ("ochenta
        al año"), que está en el trozo siguiente. Se recupera fino y se
        responde con el artículo completo.
        """
        partes = self.section_chunks(chunk)
        if not partes:
            return chunk.text, chunk.cite()

        # Si el artículo no cabe entero (el 37 del ET ocupa cuatro páginas), la
        # ventana se centra en la parte que casó con la pregunta y crece hacia
        # los lados. Cortar por el principio dejaba fuera justo el apartado
        # buscado —así se perdía el permiso de lactancia, que va al final.
        if sum(len(p.text) for p in partes) > max_chars:
            centro = next((i for i, p in enumerate(partes) if p.id == chunk.id), 0)
            izq = der = centro
            total = len(partes[centro].text)
            while True:
                creció = False
                if der + 1 < len(partes) and total + len(partes[der + 1].text) <= max_chars:
                    der += 1
                    total += len(partes[der].text)
                    creció = True
                if izq - 1 >= 0 and total + len(partes[izq - 1].text) <= max_chars:
                    izq -= 1
                    total += len(partes[izq].text)
                    creció = True
                if not creció:
                    break
            partes = partes[izq:der + 1]

        # los trozos solapan: se pega solo lo nuevo de cada uno
        texto = partes[0].text
        for parte in partes[1:]:
            cuerpo = parte.text.split("\n", 1)[-1]  # sin repetir el encabezado
            solape = min(len(texto), len(cuerpo), OVERLAP_CHARS * 2)
            corte = 0
            for n in range(solape, 20, -1):
                if texto.endswith(cuerpo[:n]):
                    corte = n
                    break
            texto += cuerpo[corte:]

        paginas = sorted({p.page for p in partes})
        rango = f"pág. {paginas[0]}" if len(paginas) == 1 else f"págs. {paginas[0]}-{paginas[-1]}"
        cita = f"{chunk.doc_title} — {chunk.section} ({rango})"
        return texto, cita

    # --- búsqueda ---------------------------------------------------------

    def embed_query(self, query: str) -> np.ndarray:
        if self._embedder is None:
            self._embedder = self._embedder_for(self.model_key)
        prefix = MODELS[self.model_key]["query_prefix"]
        vec = np.array(list(self._embedder.embed([prefix + query]))[0], dtype=np.float32)
        return vec / np.linalg.norm(vec)

    def _diversify(self, ranked: list[tuple[int, float]], k: int,
                   max_per_section: int) -> list[tuple[float, Chunk]]:
        """
        Limita cuántos fragmentos del mismo artículo entran en el resultado.

        Un artículo largo se trocea en varias partes muy parecidas entre sí y,
        sin este límite, copan los k huecos: el modelo recibe cinco veces lo
        mismo y se queda sin el artículo que de verdad completaba la respuesta.
        """
        vistos: Counter = Counter()
        salida = []
        for idx, score in ranked:
            chunk = self.chunks[idx]
            clave = (chunk.doc_id, chunk.section)
            if vistos[clave] >= max_per_section:
                continue
            vistos[clave] += 1
            salida.append((float(score), chunk))
            if len(salida) == k:
                break
        return salida

    def search(self, query: str, k: int = 5, mode: str = DEFAULT_MODE,
               max_per_section: int = 2,
               extra_queries: list[str] | None = None) -> list[tuple[float, Chunk]]:
        """
        Busca los k fragmentos más relevantes.

        `extra_queries` son reformulaciones de la misma pregunta (ver
        expand_query en rag.py): se buscan también y se fusionan con la
        original, que siempre pesa más.
        """
        pool = min(max(k * 10, 50), len(self.chunks))
        consultas = [query] + list(extra_queries or [])
        rankings, pesos = [], []

        if mode in ("semantic", "hybrid"):
            for i, consulta in enumerate(consultas):
                sims = self.vectors @ self.embed_query(consulta)
                if i == 0:
                    sims_orig = sims
                rankings.append(np.argsort(-sims)[:pool].tolist())
                pesos.append(SEMANTIC_WEIGHT if i == 0 else EXPANSION_WEIGHT)
        if mode in ("keyword", "hybrid"):
            bm = self.bm25.scores(" ".join(consultas))
            rankings.append([int(i) for i in np.argsort(-bm)[:pool] if bm[i] > 0])
            pesos.append(KEYWORD_WEIGHT)

        # una sola lista: se puede usar la puntuación real en vez de la fusión
        if len(rankings) == 1:
            puntuacion = sims_orig if mode == "semantic" else bm
            ranked = [(i, puntuacion[i]) for i in rankings[0]]
        else:
            fused = _rrf(rankings, weights=pesos)
            ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)

        return self._diversify(ranked, k, max_per_section)
