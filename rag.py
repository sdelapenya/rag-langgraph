#!/usr/bin/env python3
"""
RAG sobre documentos: recupera los fragmentos relevantes y responde citando
de qué documento, artículo y página sale cada afirmación.

    python3 rag.py "¿cuántas horas extra al año se pueden hacer?"
    python3 rag.py                       # modo interactivo
    python3 rag.py --pdf contrato.pdf "¿cuál es el plazo de entrega?"

Necesita un índice creado con `python3 ingest.py` (o usar --pdf, que indexa al
vuelo sin guardar nada).
"""
from __future__ import annotations

import argparse
import logging
import os
import random
import re
import time
from functools import lru_cache
from pathlib import Path

from groq import Groq, RateLimitError

from chunking import Chunk, chunk_pdf
from store import DEFAULT_MODE, DEFAULT_MODEL, MODELS, Index

log = logging.getLogger(__name__)

# gpt-oss-20b responde en ~0,7 s. gpt-oss-120b acierta algo más pero es un
# modelo de razonamiento y tardaba entre 6 y 25 s con este contexto, demasiado
# para una demo en la que alguien escribe y espera; llama-3.3-70b va igual de
# rápido y acierta un poco más, pero agota antes la cuota diaria de la cuenta.
MODEL_LLM = os.getenv("RAG_LLM", "openai/gpt-oss-20b")
MODEL_REESCRITURA = os.getenv("RAG_LLM_QUERY", "llama-3.1-8b-instant")

# Respaldo cuando Groq agota la cuota diaria: el plan gratis de Gemini cuenta
# peticiones al día (1.000), no tokens, así que aguanta un orden de magnitud
# más de preguntas que los 200K tokens/día de gpt-oss-20b.
# OJO: el tier gratis de Gemini entrena con lo que se le manda. Vale para esta
# demo (el corpus es BOE, texto público); NO vale para documentos de cliente.
MODEL_FALLBACK = os.getenv("RAG_LLM_FALLBACK", "gemini-3.5-flash-lite")

# k=3 y 2.500 caracteres por artículo dejan la pregunta en ~2,5K tokens en vez
# de los ~4,4K de k=5 con artículos de 4.000: casi el doble de preguntas por la
# misma cuota. Medido con evaluate.py, el recall y el acierto no se resienten.
TOP_K = int(os.getenv("RAG_TOP_K", "3"))
MAX_CHARS_ARTICULO = int(os.getenv("RAG_MAX_CHARS", "2500"))

REESCRITURA = """Reescribe la pregunta del usuario con el vocabulario formal que usaría el texto legal español al regular ese asunto.

- Devuelve SOLO la reformulación, en una línea, sin comillas ni explicaciones.
- Usa los términos técnicos equivalentes ("fichar" -> "registro de jornada", "me echan" -> "despido", "la luz y el ordenador" -> "gastos y medios necesarios").
- No inventes artículos ni cifras. No respondas la pregunta."""

# La versión anterior abría con tres reglas seguidas empujando a abstenerse y un
# modelo de 20B las sobreaplicaba: abstenía con el dato delante. Esta ordena el
# trabajo como procedimiento —mira primero, ríndete después— y permite contestar
# a la parte que sí está en vez de tirar la respuesta entera. La regla 3 obliga
# además a acotar las disposiciones excepcionales: sin ella, a "¿cuál es el IVA
# de las mascarillas?" contestaba "el 0 %" en vez de explicar que fue una medida
# COVID con fechas. Eso no es puntuación, es no dar un consejo peligroso.
SYSTEM = """Eres un asistente que responde preguntas basándose ÚNICAMENTE en los fragmentos de documento que se te dan.

Procedimiento, en este orden:
1. Recorre los fragmentos buscando el dato que se pide. Los fragmentos son artículos completos: el dato puede estar en cualquier punto, no solo al principio.
2. Si lo encuentras, respóndelo citando el fragmento. Que el artículo trate de un supuesto más amplio no impide usarlo.
3. Solo si tras ese repaso el dato no está en ningún fragmento, responde exactamente: "No encuentro esa información en los documentos."

Reglas:
1. No completes nunca con conocimiento propio: si el dato no está en los fragmentos, no lo deduzcas ni lo estimes.
2. La pregunta puede ser de otra materia (fiscal, pensiones, ayudas) aunque suene parecida a lo que regulan estos documentos. En ese caso corresponde decir que no lo encuentras.
3. Si el fragmento que usas es una disposición adicional o transitoria, o regula una situación excepcional, un colectivo concreto o un periodo con fechas, la respuesta tiene que decirlo con esas palabras: para qué supuesto se aprobó, desde cuándo y hasta cuándo. Nunca presentes como regla general lo que el texto acota.
4. Si la pregunta tiene varias partes y el texto responde solo a algunas, responde a esas y di de cuáles no habla. No uses la frase de arriba si estás contestando a alguna parte.
5. Cita siempre la fuente con el número del fragmento entre corchetes, así: [1]. Cada afirmación relevante lleva su cita.
6. Responde en español, de forma directa y breve (2-5 frases). Sin preámbulos.

FRAGMENTOS:
{context}"""


def _load_env_key(nombre: str = "GROQ_API_KEY") -> str:
    """
    Las claves de API viven fuera del repo, nunca en el código.

    Se busca primero en el fichero que indique RAG_ENV_FILE y, si no está, en
    los sitios habituales del usuario. Formato: una línea `CLAVE=valor`.
    """
    candidatos = [Path(p).expanduser() for p in (os.getenv("RAG_ENV_FILE", ""),) if p]
    candidatos += [Path.home() / "secrets" / "rag-demo.env",
                   Path.home() / ".env"]
    for path in candidatos:
        if path.exists():
            for line in path.read_text().splitlines():
                line = line.strip()
                if line.startswith(f"{nombre}="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


REINTENTOS = int(os.getenv("RAG_REINTENTOS", "4"))


def _llamar_groq(api_key: str, model: str, system: str, user: str,
                 temperature: float, max_tokens: int) -> str:
    """
    Llama a Groq reintentando cuando corta por límite de tokens.

    El límite por minuto del plan gratuito es de 8K tokens y una pregunta gasta
    unos 2.500: tres seguidas lo agotan. El error trae el tiempo de espera
    sugerido, así que se respeta.
    """
    cliente = Groq(api_key=api_key)
    for intento in range(REINTENTOS):
        try:
            resp = cliente.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content.strip()
        except RateLimitError as err:
            sugerido = re.search(r"try again in ([\d.]+)s", str(err))
            espera = float(sugerido.group(1)) if sugerido else 2 ** intento
            # esperas largas = cuota diaria agotada, no un pico por minuto:
            # ahí no sirve de nada reintentar, toca el respaldo
            if espera > 120 or intento == REINTENTOS - 1:
                raise
            time.sleep(espera + random.uniform(0, 0.5))
    return ""


@lru_cache(maxsize=1)
def _cliente_gemini(key: str):
    """
    El cliente se guarda a propósito. Creándolo al vuelo dentro de la llamada
    (`genai.Client(...).models.generate_content(...)`) no queda referenciado por
    nadie: el recolector se lo lleva con la petición en curso, cierra el httpx
    de debajo y salta «Cannot send a request, as the client has been closed».
    """
    from google import genai
    return genai.Client(api_key=key)


def _llamar_gemini(system: str, user: str, temperature: float, max_tokens: int) -> str:
    """
    Respaldo cuando Groq se queda sin cuota. El import es perezoso a propósito:
    sin google-genai instalado o sin clave, esto lanza y la traza encadena el
    429 de Groq que lo provocó, que es el fallo que hay que leer.
    """
    from google.genai import types

    key = os.getenv("GEMINI_API_KEY") or _load_env_key("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("no hay GEMINI_API_KEY para el respaldo")

    # El razonamiento se cobra contra max_output_tokens y aquí no hace falta:
    # la respuesta son 2-5 frases citando el contexto. Sin bajarlo, el modelo
    # se gasta el presupuesto "pensando" y devuelve texto vacío. Los Gemini 3
    # lo controlan con thinking_level; los 2.5, con thinking_budget.
    pensar = (types.ThinkingConfig(thinking_level="minimal")
              if MODEL_FALLBACK.startswith("gemini-3")
              else types.ThinkingConfig(thinking_budget=0))

    resp = _cliente_gemini(key).models.generate_content(
        model=MODEL_FALLBACK,
        contents=user,
        config=types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            # margen sobre el de Groq: el plan gratis de Gemini cuenta
            # peticiones, no tokens, así que aquí no se gana nada apretando
            max_output_tokens=max(max_tokens, 1200),
            thinking_config=pensar,
        ),
    )
    return (resp.text or "").strip()


def _generar(api_key: str, model: str, system: str, user: str,
             temperature: float, max_tokens: int) -> tuple[str, str]:
    """Devuelve (respuesta, proveedor). Groq primero; Gemini si no queda cuota."""
    try:
        return _llamar_groq(api_key, model, system, user, temperature, max_tokens), model
    except RateLimitError as err:
        log.warning("Groq sin cuota (%s), pasando a %s", err.__class__.__name__, MODEL_FALLBACK)
        return _llamar_gemini(system, user, temperature, max_tokens), MODEL_FALLBACK


def expand_query(question: str, api_key: str) -> str:
    """
    Traduce la pregunta al registro del documento antes de buscar.

    Los embeddings comparan cómo suena la pregunta con cómo suena el texto, y
    una persona no pregunta "compensación de gastos por trabajo a distancia",
    pregunta "quién me paga la luz". Reescribirla con un modelo pequeño (barato
    y rápido) sube claramente el recall — está medido en evaluate.py.
    """
    try:
        texto, _ = _generar(api_key, MODEL_REESCRITURA, REESCRITURA, question,
                            temperature=0.0, max_tokens=80)
        return texto.strip('"')
    except Exception:
        return ""  # si falla, se busca solo con la pregunta original


def build_sources(index: Index, hits: list[tuple[float, Chunk]]) -> list[dict]:
    """Convierte los fragmentos recuperados en fuentes con el artículo completo."""
    fuentes = []
    for n, (score, chunk) in enumerate(hits, start=1):
        texto, cita = index.section_text(chunk, max_chars=MAX_CHARS_ARTICULO)
        fuentes.append({"n": n, "cite": cita, "score": round(float(score), 4), "text": texto})
    return fuentes


def build_context(fuentes: list[dict]) -> str:
    return "\n\n---\n\n".join(f"[{f['n']}] {f['cite']}\n{f['text']}" for f in fuentes)


def ask(question: str, fuentes: list[dict], api_key: str) -> tuple[str, str]:
    if not fuentes:
        return "No encuentro esa información en los documentos.", "-"
    return _generar(
        api_key, MODEL_LLM,
        SYSTEM.format(context=build_context(fuentes)), question,
        temperature=0.0, max_tokens=700,
    )


def answer(index: Index, question: str, k: int = TOP_K, mode: str = DEFAULT_MODE,
           api_key: str | None = None, expandir: bool = True,
           hits: list[tuple[float, Chunk]] | None = None) -> dict:
    """Devuelve la respuesta y las fuentes usadas, listo para CLI o web."""
    api_key = api_key or os.getenv("GROQ_API_KEY") or _load_env_key()
    if not api_key:
        raise SystemExit("Falta GROQ_API_KEY: ponla en el entorno o en el fichero de claves (RAG_ENV_FILE)")
    reescritura = ""
    if hits is None:
        reescritura = expand_query(question, api_key) if expandir else ""
        hits = index.search(question, k=k, mode=mode, max_per_section=1,
                            extra_queries=[reescritura] if reescritura else None)
    fuentes = build_sources(index, hits)
    respuesta, proveedor = ask(question, fuentes, api_key)
    return {
        "question": question,
        "reescritura": reescritura,
        "answer": respuesta,
        "model": proveedor,
        "sources": fuentes,
    }


def index_from_pdf(path: Path, model_key: str) -> Index:
    """Índice en memoria para un PDF suelto o una carpeta, sin guardar nada."""
    pdfs = [path] if path.is_file() else sorted(path.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No encuentro PDFs en: {path}")
    chunks = []
    for pdf in pdfs:
        print(f"  Leyendo {pdf.name}...", end=" ", flush=True)
        trozos = chunk_pdf(pdf, pdf.stem, pdf.stem.replace("-", " "))
        chunks.extend(trozos)
        print(f"{len(trozos)} fragmentos")
    return Index.build(chunks, model_key)


def print_result(result: dict) -> None:
    print(f"\n{result['answer']}\n")
    print("Fuentes:")
    for src in result["sources"]:
        print(f"  [{src['n']}] {src['cite']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pregunta a tus documentos")
    parser.add_argument("question", nargs="*", help="la pregunta (vacío = interactivo)")
    parser.add_argument("--pdf", help="indexar este PDF o carpeta al vuelo")
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=list(MODELS))
    parser.add_argument("--mode", default=DEFAULT_MODE, choices=["hybrid", "semantic", "keyword"])
    parser.add_argument("-k", type=int, default=TOP_K, help="fragmentos a recuperar")
    args = parser.parse_args()

    index = index_from_pdf(Path(args.pdf), args.model) if args.pdf else Index.load(args.model)
    meta = index.meta
    question = " ".join(args.question)

    if question:
        print_result(answer(index, question, k=args.k, mode=args.mode))
        return

    print(f"\n{meta['n_chunks']} fragmentos de: {', '.join(meta['documents'])}")
    print(f"Búsqueda: {args.mode} · modelo: {meta['model']}")
    print("Escribe tu pregunta ('salir' para terminar):\n")
    while True:
        try:
            q = input(">> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nHasta luego.")
            break
        if not q or q.lower() in ("salir", "exit", "quit"):
            break
        print_result(answer(index, q, k=args.k, mode=args.mode))
        print("-" * 60)


if __name__ == "__main__":
    main()
