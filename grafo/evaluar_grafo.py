#!/usr/bin/env python3
"""
Pasa el conjunto de evaluación del RAG por el grafo y compara con el original.

    python3 evaluar_grafo.py                     # las 24 preguntas
    python3 evaluar_grafo.py --solo-fuera        # solo las 4 sin respuesta
    python3 evaluar_grafo.py --json salida.json

Mide lo mismo que evaluate.py en ~/lab/rag/ para que los números se puedan poner
uno al lado del otro, más lo único que este grafo cambia de verdad:

  llamadas_modelo_grande — cuántas de las 24 preguntas llegaron a `generar`.
                           Cada abstención resuelta antes es una llamada menos.
  abstenciones_indebidas — preguntas CON respuesta en el corpus que el grafo
                           cortó. Es el riesgo de meter un juez delante: hay que
                           mirarlo, no solo presumir de las que corta bien.

Recall y MRR no se recalculan: la recuperación es exactamente la misma función
del RAG original, así que son los números de evaluate.py sin tocar.
"""
from __future__ import annotations

import argparse
import json
import time
import unicodedata
from pathlib import Path

import puente

from grafo import NO_SE, preguntar  # noqa: E402

PREGUNTAS = puente.RAG_DIR / "eval-preguntas.json"
REESCRITURAS = puente.RAG_DIR / "eval-reescrituras.json"


def normalizar(texto: str) -> str:
    sin_tildes = unicodedata.normalize("NFKD", texto.lower())
    return "".join(c for c in sin_tildes if not unicodedata.combining(c))


NO_SE_NORM = normalizar(NO_SE)[:30]  # "no encuentro esa informacion..."


def evaluar(preguntas: list[dict], cache: dict) -> dict:
    dentro = [p for p in preguntas if not p.get("abstenerse")]
    fuera = [p for p in preguntas if p.get("abstenerse")]
    aciertos = abstenciones = generaciones = 0
    detalle = []
    t0 = time.time()

    for p in preguntas:
        estado = preguntar(p["pregunta"], reescritura=cache.get(p["id"], ""))
        texto = normalizar(estado["respuesta"])
        dijo_no_se = NO_SE_NORM in texto
        generaciones += "generar" in estado["traza"]

        fila = {
            "id": p["id"],
            "pregunta": p["pregunta"],
            "ruta": " -> ".join(estado["traza"]),
            "similitud_max": max(estado["similitudes"], default=None),
            "veredicto": estado.get("veredicto"),
            "motivo": estado.get("motivo"),
            "respuesta": estado["respuesta"],
        }
        if p.get("abstenerse"):
            fila["abstiene"] = dijo_no_se
            abstenciones += dijo_no_se
        else:
            # se distingue quién se calló: si cortó el juez, es culpa del grafo;
            # si llegó a generar y el modelo grande dijo que no lo encontraba,
            # es el comportamiento que ya tenía el RAG original
            fila["cortada_por_juez"] = "abstenerse" in estado["traza"]
            fila["abstencion_indebida"] = dijo_no_se
            fila["acierta"] = (not dijo_no_se) and any(
                normalizar(v) in texto for v in p["respuesta_contiene"]
            )
            aciertos += fila["acierta"]
        detalle.append(fila)
        print(f"  {fila['ruta']:34} {p['id']}")

    por_juez = [f["id"] for f in detalle if f.get("cortada_por_juez")]
    por_generador = [f["id"] for f in detalle
                     if f.get("abstencion_indebida") and not f.get("cortada_por_juez")]
    return {
        "resumen": {
            "n_en_corpus": len(dentro),
            "n_fuera_corpus": len(fuera),
            "acierto_respuesta": round(aciertos / len(dentro), 3) if dentro else None,
            "abstencion_correcta": round(abstenciones / len(fuera), 3) if fuera else None,
            "llamadas_modelo_grande": f"{generaciones}/{len(preguntas)}",
            "cortadas_por_el_juez": por_juez,
            "abstenidas_al_generar": por_generador,
            "segundos": round(time.time() - t0, 1),
        },
        "detalle": detalle,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evalúa el grafo")
    parser.add_argument("--solo-fuera", action="store_true",
                        help="solo las preguntas cuya respuesta no está en el corpus")
    parser.add_argument("--ids", help="lista de ids separados por comas (para reprobar un caso concreto)")
    parser.add_argument("--json", default="eval-grafo.json", help="dónde guardar el detalle")
    args = parser.parse_args()

    preguntas = json.loads(PREGUNTAS.read_text(encoding="utf-8"))["preguntas"]
    if args.solo_fuera:
        preguntas = [p for p in preguntas if p.get("abstenerse")]
    if args.ids:
        quiere = {i.strip() for i in args.ids.split(",")}
        preguntas = [p for p in preguntas if p["id"] in quiere]
    cache = json.loads(REESCRITURAS.read_text(encoding="utf-8")) if REESCRITURAS.exists() else {}

    resultado = evaluar(preguntas, cache)
    print()
    for clave, valor in resultado["resumen"].items():
        print(f"    {clave:24} {valor}")

    Path(args.json).write_text(
        json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n  Detalle en {args.json}")


if __name__ == "__main__":
    main()
