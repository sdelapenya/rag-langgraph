"""Orquestación: abstención, montaje del contexto y resiliencia de proveedor.

Ninguno de estos tests llama a una API de verdad. Lo que se comprueba es la
lógica que rodea a la llamada, que es donde están las decisiones (cuándo no
responder, cuándo reintentar y cuándo cambiar de proveedor).
"""
from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest
from groq import RateLimitError

import rag
from rag import _load_env_key, ask, build_context, build_sources

ABSTENCION = "No encuentro esa información en los documentos."


def sin_cuota(mensaje: str = "Rate limit reached, try again in 3600.0s") -> RateLimitError:
    peticion = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    return RateLimitError(mensaje, response=httpx.Response(429, request=peticion), body=None)


def respuesta_llm(texto: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=texto))]
    )


# --- claves ----------------------------------------------------------------

def test_la_clave_se_lee_del_fichero_no_del_codigo(tmp_path, monkeypatch):
    fichero = tmp_path / "claves.env"
    fichero.write_text('GROQ_API_KEY="gsk-de-prueba"\nOTRA_COSA=1\n')
    monkeypatch.setenv("RAG_ENV_FILE", str(fichero))

    assert _load_env_key() == "gsk-de-prueba"


def test_sin_fichero_de_claves_devuelve_vacio(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_ENV_FILE", str(tmp_path / "no-existe.env"))
    monkeypatch.setattr(rag.Path, "home", staticmethod(lambda: tmp_path))

    assert _load_env_key() == ""


# --- abstención ------------------------------------------------------------

def test_sin_fuentes_se_abstiene_sin_gastar_una_llamada(monkeypatch):
    """Si la búsqueda no ha traído nada, preguntar al modelo solo puede salir
    mal: contestaría de memoria, que es justo lo que este sistema evita."""
    def no_deberia_llamarse(*_args, **_kwargs):
        raise AssertionError("no debería haberse llamado a ningún proveedor")

    monkeypatch.setattr(rag, "_generar", no_deberia_llamarse)

    assert ask("¿cuántas horas extra?", [], "clave") == (ABSTENCION, "-")


# --- contexto --------------------------------------------------------------

def test_build_context_numera_cada_fuente_para_que_el_modelo_pueda_citarla():
    fuentes = [
        {"n": 1, "cite": "Estatuto — Artículo 34 (pág. 10)", "score": 0.9,
         "text": "La jornada ordinaria."},
        {"n": 2, "cite": "Estatuto — Artículo 35 (pág. 12)", "score": 0.8,
         "text": "Las horas extraordinarias."},
    ]
    contexto = build_context(fuentes)

    assert contexto.startswith("[1] Estatuto — Artículo 34 (pág. 10)\nLa jornada ordinaria.")
    assert "[2] Estatuto — Artículo 35 (pág. 12)" in contexto
    assert "\n\n---\n\n" in contexto


def test_build_sources_manda_el_articulo_completo_no_solo_el_fragmento(indice):
    """Se recupera fino y se responde con el artículo entero: el trozo que mejor
    casa con la pregunta no tiene por qué ser el que lleva el dato."""
    fuentes = build_sources(indice, [(0.8712, indice.chunks[0])])

    assert len(fuentes) == 1
    assert fuentes[0]["n"] == 1
    assert fuentes[0]["score"] == 0.8712
    assert fuentes[0]["cite"] == "Estatuto — Artículo 34 (págs. 10-11)"
    assert "Descanso entre jornadas." in fuentes[0]["text"]


# --- resiliencia de proveedor ---------------------------------------------

def test_groq_reintenta_ante_un_pico_por_minuto(monkeypatch):
    llamadas = []

    def create(**kwargs):
        llamadas.append(kwargs)
        if len(llamadas) == 1:
            raise sin_cuota("Rate limit reached, try again in 0.5s")
        return respuesta_llm("  ochenta horas al año  ")

    monkeypatch.setattr(rag, "Groq", lambda api_key=None: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    monkeypatch.setattr(rag.time, "sleep", lambda _segundos: None)

    assert rag._llamar_groq("k", "m", "sys", "user", 0.0, 100) == "ochenta horas al año"
    assert len(llamadas) == 2


def test_groq_no_reintenta_cuando_la_cuota_es_diaria(monkeypatch):
    """Una espera de horas no es un pico por minuto: reintentar solo retrasa el
    respaldo. El umbral está en 120 s."""
    llamadas = []

    def create(**kwargs):
        llamadas.append(kwargs)
        raise sin_cuota("Rate limit reached, try again in 3600.0s")

    monkeypatch.setattr(rag, "Groq", lambda api_key=None: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    monkeypatch.setattr(rag.time, "sleep", lambda _segundos: None)

    with pytest.raises(RateLimitError):
        rag._llamar_groq("k", "m", "sys", "user", 0.0, 100)
    assert len(llamadas) == 1


def test_cuando_groq_se_queda_sin_cuota_responde_gemini(monkeypatch):
    def groq_agotado(*_args, **_kwargs):
        raise sin_cuota()

    monkeypatch.setattr(rag, "_llamar_groq", groq_agotado)
    monkeypatch.setattr(rag, "_llamar_gemini", lambda *_a, **_k: "respuesta del respaldo")

    texto, proveedor = rag._generar("k", "modelo-groq", "sys", "user", 0.0, 100)

    assert texto == "respuesta del respaldo"
    assert proveedor == rag.MODEL_FALLBACK


def test_si_falla_la_reescritura_se_busca_con_la_pregunta_original(monkeypatch):
    """La reescritura mejora el recall, pero es un extra: si el modelo pequeño
    falla, la búsqueda tiene que seguir funcionando."""
    def revienta(*_args, **_kwargs):
        raise RuntimeError("la API no responde")

    monkeypatch.setattr(rag, "_generar", revienta)

    assert rag.expand_query("¿quién me paga la luz?", "clave") == ""


def test_la_reescritura_llega_limpia_de_comillas(monkeypatch):
    monkeypatch.setattr(
        rag, "_generar",
        lambda *_a, **_k: ('"compensación de gastos por trabajo a distancia"', "m"),
    )

    assert rag.expand_query("¿quién me paga la luz?", "clave") == (
        "compensación de gastos por trabajo a distancia"
    )
