"""Índice: BM25, fusión RRF, reconstrucción del artículo y persistencia."""
from __future__ import annotations

import numpy as np
import pytest
from conftest import chunk

from store import BM25, Index, _rrf, tokenize

# --- tokenización ----------------------------------------------------------

def test_tokenize_baja_a_minusculas_y_quita_vacias():
    assert tokenize("El plazo DE entrega y la fecha") == ["plazo", "entrega", "fecha"]


# --- BM25 ------------------------------------------------------------------

CORPUS = [
    "el despido improcedente da derecho a indemnización",
    "la jornada ordinaria máxima es de cuarenta horas",
    "las horas extraordinarias no superarán ochenta al año",
]


def test_bm25_solo_puntua_documentos_que_contienen_el_termino():
    puntuaciones = BM25([tokenize(t) for t in CORPUS]).scores("horas extraordinarias")
    assert puntuaciones[0] == 0.0                      # no menciona ninguno
    assert puntuaciones[2] > puntuaciones[1] > 0.0     # dos términos gana a uno


def test_bm25_ignora_terminos_que_no_estan_en_el_corpus():
    bm = BM25([tokenize("jornada ordinaria")])
    assert bm.scores("criptomonedas").tolist() == [0.0]


def test_bm25_con_corpus_vacio_no_revienta():
    assert BM25([]).scores("lo que sea").tolist() == []


# --- fusión RRF ------------------------------------------------------------

def test_rrf_sin_pesos_empata_dos_listas_simetricas():
    fusionado = _rrf([[1, 2], [2, 1]])
    assert fusionado[1] == fusionado[2]


def test_rrf_deja_ganar_a_la_lista_que_mas_pesa():
    """La semántica acierta más que BM25 en este corpus, así que pesa más:
    con pesos iguales, la lista mala arrastraría a la buena."""
    fusionado = _rrf([[10, 20], [20, 10]], weights=[1.0, 0.4])
    assert fusionado[10] > fusionado[20]


# --- reconstrucción del artículo ------------------------------------------

def test_section_chunks_devuelve_el_articulo_entero_y_en_orden(indice):
    partes = indice.section_chunks(indice.chunks[1])
    assert [p.part for p in partes] == [1, 2]


def test_section_text_cita_el_rango_de_paginas(indice):
    texto, cita = indice.section_text(indice.chunks[0])
    assert cita == "Estatuto — Artículo 34 (págs. 10-11)"
    # el hit era la parte 1, pero la respuesta lleva también la 2
    assert "Descanso entre jornadas." in texto


def test_section_text_no_repite_el_encabezado_en_cada_parte(indice):
    texto, _ = indice.section_text(indice.chunks[0])
    assert texto.count("Artículo 34") == 1


def test_section_text_con_una_sola_pagina_usa_singular(indice):
    _, cita = indice.section_text(indice.chunks[2])
    assert cita == "Estatuto — Artículo 35 (pág. 12)"


def test_section_text_no_duplica_el_texto_solapado():
    """Los fragmentos se solapan a propósito; al recomponer el artículo hay que
    pegar solo lo nuevo o el modelo recibe el mismo párrafo dos veces."""
    comun = "y el descanso mínimo será de doce horas entre jornada y jornada"
    primera = chunk(0, "Artículo 34", 1, f"Artículo 34\nLa jornada ordinaria {comun}",
                    part=1, n_parts=2)
    segunda = chunk(1, "Artículo 34", 2, f"Artículo 34\n{comun} salvo pacto en contrario",
                    part=2, n_parts=2)
    indice = Index([primera, segunda], np.eye(2, dtype=np.float32), "minilm")

    texto, _ = indice.section_text(primera)
    assert texto.count(comun) == 1
    assert texto.endswith("salvo pacto en contrario")


def test_section_text_recorta_alrededor_del_fragmento_que_caso():
    """Si el artículo no cabe entero, la ventana se centra en la parte que casó
    con la pregunta. Cortando por el principio se perdía justo el apartado
    buscado, que en artículos largos suele ir al final."""
    partes = [
        chunk(i, "Artículo 37", 1, f"Artículo 37\n{'x' * 900} apartado {i}",
              part=i + 1, n_parts=6)
        for i in range(6)
    ]
    indice = Index(partes, np.eye(6, dtype=np.float32), "minilm")

    texto, _ = indice.section_text(partes[5], max_chars=2500)
    assert "apartado 5" in texto      # el que casó, aunque sea el último
    assert "apartado 0" not in texto  # se ha recortado por el otro lado
    assert len(texto) <= 2500


# --- diversificación -------------------------------------------------------

def test_diversify_no_deja_que_un_articulo_cope_los_huecos(indice):
    ranked = [(0, 0.9), (1, 0.8), (2, 0.7)]  # las dos primeras son del art. 34
    salida = indice._diversify(ranked, k=3, max_per_section=1)
    assert [c.section for _, c in salida] == ["Artículo 34", "Artículo 35"]


def test_diversify_admite_varias_partes_si_se_permiten(indice):
    ranked = [(0, 0.9), (1, 0.8), (2, 0.7)]
    assert len(indice._diversify(ranked, k=3, max_per_section=2)) == 3


def test_diversify_corta_en_k(indice):
    ranked = [(0, 0.9), (1, 0.8), (2, 0.7)]
    assert len(indice._diversify(ranked, k=1, max_per_section=2)) == 1


# --- persistencia ----------------------------------------------------------

def test_guardar_y_cargar_conserva_fragmentos_y_vectores(indice, tmp_path):
    indice.save(tmp_path)
    recargado = Index.load("minilm", directory=tmp_path)

    assert [c.id for c in recargado.chunks] == [c.id for c in indice.chunks]
    assert [c.page for c in recargado.chunks] == [10, 11, 12]
    assert np.array_equal(recargado.vectors, indice.vectors)
    assert recargado.meta["n_chunks"] == 3


def test_cargar_sin_indice_explica_como_crearlo(tmp_path):
    with pytest.raises(FileNotFoundError, match="ingest.py"):
        Index.load("minilm", directory=tmp_path)
