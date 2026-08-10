#!/usr/bin/env python3
"""
Mide si el RAG funciona, en vez de suponerlo.

    python3 evaluate.py --solo-recuperacion        # sin llamadas al LLM (gratis)
    python3 evaluate.py --solo-recuperacion --comparar
    python3 evaluate.py                            # completo: recuperación + respuesta

Métricas:
  recall@k  — de las preguntas con respuesta en el corpus, en cuántas aparece
              el artículo correcto entre los k fragmentos recuperados.
  MRR       — 1/posición del primer fragmento correcto (premia acertar arriba).
  acierto   — el dato esperado aparece en la respuesta del modelo.
  abstención— ante preguntas cuya respuesta NO está en los documentos, el
              modelo dice que no la encuentra en vez de inventarla. Es la
              métrica que decide si esto se puede enseñar a un cliente.
"""
from __future__ import annotations

import argparse
import json
import time
import unicodedata
from pathlib import Path

from rag import TOP_K, _load_env_key, answer, expand_query
from store import MODELS, Index

PREGUNTAS = Path("eval-preguntas.json")
CACHE_REESCRITURAS = Path("eval-reescrituras.json")
NO_SE = "no encuentro esa informacion"


def reescrituras(preguntas: list[dict], api_key: str) -> dict[str, str]:
    """
    Reescribe cada pregunta una sola vez y lo guarda en disco.

    Sin caché, comparar seis configuraciones dispararía seis veces las mismas
    llamadas y además metería ruido: la reescritura no es determinista del todo.
    """
    cache = json.loads(CACHE_REESCRITURAS.read_text(encoding="utf-8")) if CACHE_REESCRITURAS.exists() else {}
    nuevas = False
    for p in preguntas:
        if p["id"] not in cache:
            cache[p["id"]] = expand_query(p["pregunta"], api_key)
            nuevas = True
    if nuevas:
        CACHE_REESCRITURAS.write_text(
            json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return cache


def normalizar(texto: str) -> str:
    sin_tildes = unicodedata.normalize("NFKD", texto.lower())
    return "".join(c for c in sin_tildes if not unicodedata.combining(c))


def seccion_id(chunk) -> str:
    return f"{chunk.doc_id}:{chunk.section}"


def acierta_seccion(chunk, esperadas: list[str]) -> bool:
    sid = seccion_id(chunk)
    return any(sid.startswith(e) for e in esperadas)


def evaluar(index: Index, preguntas: list[dict], k: int, mode: str,
            con_llm: bool, expandir: bool = False,
            max_per_section: int = 1) -> dict:
    api_key = _load_env_key() if (con_llm or expandir) else None
    cache = reescrituras(preguntas, api_key) if expandir else {}
    en_corpus = [p for p in preguntas if not p.get("abstenerse")]
    fuera = [p for p in preguntas if p.get("abstenerse")]

    recall, rangos, aciertos, abstenciones, detalle = 0, [], 0, 0, []
    t0 = time.time()

    for p in preguntas:
        extra = [cache[p["id"]]] if cache.get(p["id"]) else None
        hits = index.search(p["pregunta"], k=k, mode=mode,
                            max_per_section=max_per_section, extra_queries=extra)
        fila = {"id": p["id"], "pregunta": p["pregunta"]}

        if not p.get("abstenerse"):
            posiciones = [
                i for i, (_s, c) in enumerate(hits, start=1)
                if acierta_seccion(c, p["secciones"])
            ]
            fila["recuperado"] = bool(posiciones)
            fila["rango"] = posiciones[0] if posiciones else None
            fila["top1"] = seccion_id(hits[0][1]) if hits else "-"
            recall += bool(posiciones)
            rangos.append(1 / posiciones[0] if posiciones else 0.0)

        if con_llm:
            res = answer(index, p["pregunta"], k=k, mode=mode, api_key=api_key, hits=hits)
            texto = normalizar(res["answer"])
            fila["respuesta"] = res["answer"]
            if p.get("abstenerse"):
                fila["abstiene"] = NO_SE in texto
                abstenciones += fila["abstiene"]
            else:
                fila["acierta"] = any(
                    normalizar(v) in texto for v in p["respuesta_contiene"]
                ) and NO_SE not in texto
                aciertos += fila["acierta"]

        detalle.append(fila)

    resumen = {
        "modelo": MODELS[index.model_key]["name"],
        "modo": mode,
        "reescritura": expandir,
        "k": k,
        "n_en_corpus": len(en_corpus),
        "n_fuera_corpus": len(fuera),
        f"recall@{k}": round(recall / len(en_corpus), 3),
        "mrr": round(sum(rangos) / len(rangos), 3),
        "segundos": round(time.time() - t0, 1),
    }
    if con_llm:
        resumen["acierto_respuesta"] = round(aciertos / len(en_corpus), 3)
        resumen["abstencion_correcta"] = round(abstenciones / len(fuera), 3) if fuera else None
    return {"resumen": resumen, "detalle": detalle}


def imprimir(resultado: dict, verbose: bool) -> None:
    r = resultado["resumen"]
    print(f"\n  modelo: {r['modelo']}  ·  modo: {r['modo']}"
          f"  ·  reescritura: {'sí' if r['reescritura'] else 'no'}  ·  k={r['k']}")
    for clave, valor in r.items():
        if clave not in ("modelo", "modo", "k", "reescritura"):
            print(f"    {clave:22} {valor}")
    if not verbose:
        return
    print("\n  Fallos:")
    hubo = False
    for fila in resultado["detalle"]:
        fallo_rec = "recuperado" in fila and not fila["recuperado"]
        fallo_resp = fila.get("acierta") is False or fila.get("abstiene") is False
        if fallo_rec or fallo_resp:
            hubo = True
            print(f"    [{fila['id']}] {fila['pregunta']}")
            if fallo_rec:
                print(f"        no recuperó el artículo esperado (top1: {fila['top1']})")
            if fallo_resp:
                print(f"        respuesta: {fila.get('respuesta', '')[:160]}")
    if not hubo:
        print("    ninguno")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evalúa el RAG")
    parser.add_argument("--solo-recuperacion", action="store_true",
                        help="sin llamadas al LLM: rápido y gratis")
    parser.add_argument("--comparar", action="store_true",
                        help="prueba todos los modos de búsqueda (y modelos indexados)")
    parser.add_argument("--expandir", action="store_true",
                        help="reescribir la pregunta al registro del documento antes de buscar")
    parser.add_argument("--max-por-articulo", type=int, default=1,
                        help="tope de fragmentos del mismo artículo en el resultado")
    parser.add_argument("--model", default=None, choices=list(MODELS))
    parser.add_argument("--mode", default="semantic", choices=["hybrid", "semantic", "keyword"])
    parser.add_argument("-k", type=int, default=TOP_K,
                        help="fragmentos recuperados (por defecto, el de la demo)")
    parser.add_argument("--json", help="guardar el resultado completo en este fichero")
    parser.add_argument("-v", "--verbose", action="store_true", help="listar los fallos")
    args = parser.parse_args()

    preguntas = json.loads(PREGUNTAS.read_text(encoding="utf-8"))["preguntas"]
    con_llm = not args.solo_recuperacion

    if args.comparar:
        modelos = [m for m in MODELS if (Path("index") / m / "vectors.npy").exists()]
        combinaciones = [
            (m, modo, exp)
            for m in modelos
            for modo in ("keyword", "semantic", "hybrid")
            for exp in (False, True)
        ]
    else:
        combinaciones = [(args.model, args.mode, args.expandir)]

    salida = []
    for model_key, modo, expandir in combinaciones:
        index = Index.load(model_key) if model_key else Index.load()
        resultado = evaluar(index, preguntas, args.k, modo, con_llm, expandir,
                            args.max_por_articulo)
        imprimir(resultado, args.verbose)
        salida.append(resultado)

    if args.json:
        Path(args.json).write_text(
            json.dumps(salida, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\nDetalle guardado en {args.json}")


if __name__ == "__main__":
    main()
