#!/usr/bin/env python3
"""
¿Llega la respuesta al generador? Diagnóstico sin LLM.

El acierto mezcla dos fallos muy distintos: que el sistema **no le mande** la
respuesta al modelo, y que se la mande y aun así falle. Este script separa los
dos: comprueba si alguna de las cadenas de `respuesta_contiene` aparece en el
fragmento de la sección que debería tenerla.

No llama a ninguna API —las reescrituras salen de la caché en disco—, así que es
determinista, gratis y se puede correr tantas veces como haga falta.

    python3 tools/contexto_contiene.py                 # los dos conjuntos
    python3 tools/contexto_contiene.py --recortes 0 600

Mide además el efecto de mandar la cabecera del artículo por delante de la
ventana centrada: `--recortes` toma los caracteres de cabecera a probar (`0` es
el comportamiento actual, sin cabecera; `-1`, la cabecera entera).

⚠️ Mirar si la respuesta está en el contexto **entero** engaña: en
`despido-objetivo` la frase colaba desde una disposición transitoria distinta
mientras el artículo bueno llegaba truncado. Por eso aquí solo cuenta el texto
de las secciones esperadas.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
os.chdir(RAIZ)

import rag  # noqa: E402
from evaluate import normalizar  # noqa: E402
from store import DEFAULT_MODEL, Index  # noqa: E402

CONJUNTOS = {
    "oficial": ("eval-preguntas.json", "eval-reescrituras.json"),
    "holdout": ("eval-preguntas-holdout.json", "eval-reescrituras-holdout.json"),
}
_section_text = Index.section_text


def con_cabecera(n_chars: int):
    """Devuelve un `section_text` que antepone los primeros `n_chars` del artículo."""
    def section_text(self, chunk, max_chars=4000):
        texto, cita = _section_text(self, chunk, max_chars=max_chars)
        if n_chars == 0:
            return texto, cita
        partes = self.section_chunks(chunk)
        if not partes:
            return texto, cita
        cabeza = partes[0].text if n_chars < 0 else partes[0].text[:n_chars]
        if normalizar(cabeza[:120]) in normalizar(texto):
            return texto, cita  # la ventana ya empieza por el principio
        return f"{cabeza}\n[…]\n{texto}", cita
    return section_text


def medir(ix: Index, preguntas: list[dict], cache: dict) -> tuple[int, int, int, list[str]]:
    """(respuestas que llegan, preguntas evaluables, caracteres de contexto, fallos)."""
    llegan, evaluables, chars, fallan = 0, 0, 0, []
    for p in preguntas:
        if p.get("abstenerse"):
            continue  # no hay respuesta que buscar
        extra = cache.get(p["id"], "")
        hits = ix.search(p["pregunta"], k=rag.TOP_K, mode="semantic",
                         max_per_section=1, extra_queries=[extra] if extra else None)
        fuentes = rag.build_sources(ix, hits)
        chars += len(rag.build_context(fuentes))
        # build_sources devuelve una fuente por hit y en el mismo orden
        buenas = [f for f, (_s, c) in zip(fuentes, hits, strict=True)
                  if any(f"{c.doc_id}:{c.section}".startswith(e) for e in p["secciones"])]
        if not buenas:
            continue  # no se recuperó: eso es la búsqueda, no la ventana
        evaluables += 1
        ok = any(normalizar(v) in normalizar(f["text"])
                 for f in buenas for v in p["respuesta_contiene"])
        llegan += ok
        if not ok:
            fallan.append(p["id"])
    return llegan, evaluables, chars, fallan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recortes", nargs="+", type=int, default=[0, -1],
                        help="caracteres de cabecera a probar (0 = actual, -1 = entera)")
    parser.add_argument("--modelo", default=os.getenv("RAG_MODEL", DEFAULT_MODEL))
    args = parser.parse_args()

    ix = Index.load(args.modelo)
    datos = {}
    for nombre, (f_preg, f_cache) in CONJUNTOS.items():
        if not Path(f_preg).exists():
            continue
        cache = json.loads(Path(f_cache).read_text(encoding="utf-8")) if Path(f_cache).exists() else {}
        datos[nombre] = (json.loads(Path(f_preg).read_text(encoding="utf-8"))["preguntas"], cache)

    base: dict[str, int] = {}
    for n_chars in args.recortes:
        Index.section_text = con_cabecera(n_chars)
        etiqueta = {0: "sin cabecera", -1: "cabecera entera"}.get(n_chars, f"cabecera {n_chars}")
        partes = []
        for nombre, (preguntas, cache) in datos.items():
            llegan, evaluables, chars, fallan = medir(ix, preguntas, cache)
            base.setdefault(nombre, chars)
            coste = round(100 * chars / base[nombre] - 100)
            partes.append(f"{nombre}: {llegan}/{evaluables} llegan, contexto {coste:+d} %"
                          + (f" (falla {', '.join(fallan)})" if fallan else ""))
        print(f"{etiqueta:16} | " + " | ".join(partes))

    Index.section_text = _section_text


if __name__ == "__main__":
    main()
