#!/usr/bin/env python3
"""
El RAG de ~/lab/rag/ reescrito como grafo de LangGraph.

    python3 grafo.py "¿cuántas horas extra al año se pueden hacer?"
    python3 grafo.py "¿cuál es el tipo general del IVA?" -v
    python3 grafo.py --mermaid

Qué cambia respecto del RAG original: allí, decidir si el contexto vale es una
regla dentro del prompt de generación — el modelo grande recibe los fragmentos
siempre y a veces contesta "no lo encuentro". Aquí esa decisión es un nodo
propio con un modelo pequeño, y la abstención es una arista del grafo:

    recuperar -> evaluar -> generar     (el contexto responde la pregunta)
                        -> abstenerse   (no responde)

Ventaja concreta, no decorativa: cuando la respuesta no está en el corpus no se
llega a llamar al modelo grande, y el motivo de la abstención queda escrito en
el estado en vez de escondido en un prompt.
"""
from __future__ import annotations

import argparse
import json
import os
from functools import lru_cache
from operator import add
from typing import Annotated, Literal, TypedDict

import puente  # noqa: F401  — deja ~/lab/rag/ importable; tiene que ir el primero
from langgraph.graph import END, START, StateGraph

# El RAG existente. `_generar` es privado, sí: es la que encadena Groq -> Gemini
# cuando se agota la cuota diaria, y reimplementar ese respaldo aquí sería
# duplicar la única parte que ya está probada en producción.
from rag import (  # noqa: E402
    MODEL_LLM,
    MODEL_REESCRITURA,
    TOP_K,
    _generar,
    _load_env_key,
    ask,
    build_context,
    build_sources,
    expand_query,
)
from store import DEFAULT_MODE, Index  # noqa: E402

NO_SE = "No encuentro esa información en los documentos."

# Modelo juez: el mismo pequeño que reescribe la pregunta. Juzgar si un texto
# contiene un dato es tarea de clasificación, no de redacción; gpt-oss-20b aquí
# costaría lo mismo que responder y quitaría todo el sentido al filtro.
MODEL_JUEZ = os.getenv("RAG_LLM_JUEZ", MODEL_REESCRITURA)

# Suelo de similitud coseno: por debajo se abstiene sin consultar al juez.
#
# Es un suelo, no un discriminador, y eso está medido: calibrar_umbral.py mostró
# que con e5 las 20 preguntas del corpus caen entre 0,828 y 0,895 y las 4 de
# fuera entre 0,833 y 0,862 — se solapan, no hay umbral que las separe (e5
# comprime todos los cosenos en una franja estrecha). Así que el filtro barato
# se queda solo para lo que no tiene nada que ver con el corpus, por debajo de
# todo lo observado, y el trabajo fino lo hace el nodo juez. Ver README.
UMBRAL = float(os.getenv("RAG_UMBRAL_SIMILITUD", "0.80"))

JUEZ = """Decides si unos fragmentos de documento permiten responder a una pregunta. NO respondes la pregunta.

Reglas:
1. Contesta SOLO una línea con este formato: "SI: <motivo>" o "NO: <motivo>". El motivo, menos de 12 palabras.
2. La pregunta viene en lenguaje corriente y el documento está en lenguaje formal. Si el texto regula el asunto por el que se pregunta, es SI aunque lo diga con otras palabras: "fichar" es "registro horario", "quién paga la luz" es "compensación de gastos", "volver a la oficina" es "reversibilidad".
3. El dato pedido cuenta aunque venga en otra unidad o formato (un porcentaje donde preguntan días, "treinta" donde preguntan un número).
4. Es NO cuando preguntan un dato concreto —una cifra, un plazo, un importe— y ese dato no está en el texto, por mucho que el tema sí aparezca.
5. Es NO cuando la pregunta es de otra materia (fiscal, pensiones, prestaciones, trámites del SEPE) aunque suene parecida a lo que regulan los fragmentos.
6. No uses conocimiento propio: solo lo que hay escrito en los fragmentos.

FRAGMENTOS:
{context}"""

PREGUNTA_JUEZ = """Pregunta: {pregunta}
La misma pregunta en el vocabulario del documento: {reescritura}"""


class Estado(TypedDict, total=False):
    """
    Lo que viaja por el grafo.

    `traza` lleva un reductor (`add`): cada nodo devuelve su nombre en una lista
    y LangGraph las concatena, así que al terminar el estado dice por dónde pasó
    la pregunta. Es lo que se enseña con -v.
    """

    pregunta: str
    reescritura: str
    fuentes: list[dict]
    similitudes: list[float]
    suficiente: bool
    motivo: str
    veredicto: str
    respuesta: str
    modelo: str
    traza: Annotated[list[str], add]


@lru_cache(maxsize=1)
def _indice() -> Index:
    """El índice pesa ~1 GB con e5: se carga una vez por proceso."""
    return Index.load()


@lru_cache(maxsize=1)
def _api_key() -> str:
    clave = os.getenv("GROQ_API_KEY") or _load_env_key()
    if not clave:
        raise SystemExit("Falta GROQ_API_KEY: ponla en el entorno o en el fichero de claves (RAG_ENV_FILE)")
    return clave


def _similitudes(index: Index, pregunta: str, hits: list) -> list[float]:
    """
    Similitud coseno de cada fragmento con la pregunta original.

    No sirve la puntuación que devuelve `index.search()`: cuando hay reescritura
    fusiona dos rankings con RRF y salen números de otra escala (0,0x), que no
    se pueden comparar entre preguntas. El coseno sí es estable, y por eso se
    le puede poner un umbral medido.
    """
    fila = {chunk.id: i for i, chunk in enumerate(index.chunks)}
    consulta = index.embed_query(pregunta)
    return [round(float(index.vectors[fila[c.id]] @ consulta), 4) for _s, c in hits]


# --------------------------------------------------------------------- nodos


def recuperar(estado: Estado) -> Estado:
    """Reescribe la pregunta, busca en el índice y arma las fuentes."""
    index, pregunta = _indice(), estado["pregunta"]
    # si la reescritura viene ya dada (la evaluación la trae cacheada), no se
    # vuelve a pedir al modelo: es la llamada más fácil de ahorrar
    reescritura = estado.get("reescritura")
    if reescritura is None:
        reescritura = expand_query(pregunta, _api_key())

    hits = index.search(
        pregunta,
        k=TOP_K,
        mode=DEFAULT_MODE,
        max_per_section=1,
        extra_queries=[reescritura] if reescritura else None,
    )
    return {
        "reescritura": reescritura,
        "fuentes": build_sources(index, hits),
        "similitudes": _similitudes(index, pregunta, hits),
        "traza": ["recuperar"],
    }


def evaluar(estado: Estado) -> Estado:
    """
    ¿Responden estos fragmentos a la pregunta? Dos filtros, de barato a caro.

    Primero la similitud, que ya está calculada y no cuesta nada, pero solo caza
    lo que ni siquiera roza el corpus (ver UMBRAL). El caso difícil —fragmentos
    que hablan del tema y no traen el dato— es el del modelo juez.
    """
    fuentes = estado.get("fuentes") or []
    if not fuentes:
        return {"suficiente": False, "motivo": "la búsqueda no devolvió fragmentos",
                "traza": ["evaluar"]}

    mejor = max(estado["similitudes"])
    if mejor < UMBRAL:
        return {
            "suficiente": False,
            "motivo": f"nada suficientemente parecido en el corpus (similitud {mejor:.2f} < {UMBRAL:.2f})",
            "traza": ["evaluar"],
        }

    # al juez se le da también la reescritura, que ya está hecha y viene en el
    # registro del documento: sin ella confundía "no lo dice" con "no lo dice
    # con esas palabras" y tumbaba preguntas que sí tenían respuesta (5 de 20).
    reescritura = estado.get("reescritura") or "(no disponible)"
    veredicto, _proveedor = _generar(
        _api_key(), MODEL_JUEZ,
        JUEZ.format(context=build_context(fuentes)),
        PREGUNTA_JUEZ.format(pregunta=estado["pregunta"], reescritura=reescritura),
        temperature=0.0, max_tokens=60,
    )
    linea = veredicto.strip().lstrip("*# ").replace("Í", "I")
    # ante una respuesta rara del juez se tira hacia adelante: preferimos gastar
    # una generación de más a abstenernos con una pregunta que sí tenía respuesta
    suficiente = not linea.upper().startswith("NO")
    motivo = linea.split(":", 1)[1].strip() if ":" in linea else linea
    return {"suficiente": suficiente, "motivo": motivo or "-",
            "veredicto": linea, "traza": ["evaluar"]}


def generar(estado: Estado) -> Estado:
    """Redacta la respuesta con citas: es el `ask()` del RAG original."""
    respuesta, proveedor = ask(estado["pregunta"], estado["fuentes"], _api_key())
    return {"respuesta": respuesta, "modelo": proveedor, "traza": ["generar"]}


def abstenerse(estado: Estado) -> Estado:
    """
    Cierra sin llamar al modelo grande.

    La frase es literalmente la misma que devuelve el RAG original, para que
    evaluate.py la cuente igual y los dos números sean comparables.
    """
    return {"respuesta": NO_SE, "modelo": "-", "traza": ["abstenerse"]}


def decidir(estado: Estado) -> Literal["generar", "abstenerse"]:
    return "generar" if estado.get("suficiente") else "abstenerse"


# --------------------------------------------------------------------- grafo


@lru_cache(maxsize=1)
def construir():
    grafo = StateGraph(Estado)
    grafo.add_node("recuperar", recuperar)
    grafo.add_node("evaluar", evaluar)
    grafo.add_node("generar", generar)
    grafo.add_node("abstenerse", abstenerse)

    grafo.add_edge(START, "recuperar")
    grafo.add_edge("recuperar", "evaluar")
    grafo.add_conditional_edges("evaluar", decidir,
                                {"generar": "generar", "abstenerse": "abstenerse"})
    grafo.add_edge("generar", END)
    grafo.add_edge("abstenerse", END)
    return grafo.compile()


def preguntar(pregunta: str, reescritura: str | None = None) -> Estado:
    """Una pregunta de punta a punta. `reescritura` evita una llamada si ya se tiene."""
    entrada: Estado = {"pregunta": pregunta}
    if reescritura is not None:
        entrada["reescritura"] = reescritura
    return construir().invoke(entrada)


def imprimir(estado: Estado, verbose: bool) -> None:
    print(f"\n{estado['respuesta']}\n")
    if estado.get("suficiente"):
        print("Fuentes:")
        for f in estado["fuentes"]:
            print(f"  [{f['n']}] {f['cite']}")
    else:
        print(f"Motivo de la abstención: {estado.get('motivo', '-')}")
    if not verbose:
        return
    print(f"\n  ruta        {' -> '.join(estado.get('traza', []))}")
    print(f"  reescritura {estado.get('reescritura') or '(ninguna)'}")
    print(f"  similitudes {estado.get('similitudes')}  (umbral {UMBRAL})")
    print(f"  modelos     juez={MODEL_JUEZ}  respuesta={estado.get('modelo')}")
    if not estado.get("suficiente"):
        print(f"  ahorrado    una llamada a {MODEL_LLM}")


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG como grafo de LangGraph")
    parser.add_argument("pregunta", nargs="*")
    parser.add_argument("--mermaid", action="store_true",
                        help="imprime el grafo en Mermaid y sale (no carga el índice)")
    parser.add_argument("--json", action="store_true", help="estado final en JSON")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="ruta seguida, similitudes y modelos usados")
    args = parser.parse_args()

    if args.mermaid:
        print(construir().get_graph().draw_mermaid())
        return

    pregunta = " ".join(args.pregunta).strip()
    if not pregunta:
        parser.error("dime una pregunta (o usa --mermaid)")

    estado = preguntar(pregunta)
    if args.json:
        print(json.dumps(estado, ensure_ascii=False, indent=2))
    else:
        imprimir(estado, args.verbose)


if __name__ == "__main__":
    main()
