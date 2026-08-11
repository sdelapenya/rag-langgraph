"""Troceado: lo que se comprueba aquí es que cada fragmento sepa de dónde sale.

Si el troceado pierde la página o parte un artículo por la mitad de una frase,
la cita que se enseña bajo la respuesta deja de ser verificable, que es el
único motivo por el que este RAG existe.
"""
from __future__ import annotations

from chunking import (
    _SECTION_HEAD,
    MAX_CHARS,
    Chunk,
    _page_map,
    _split_long,
    _split_sections,
    clean_page,
)

RELLENO = "".join(
    f"Frase número {i} con texto de sobra para ocupar sitio. " for i in range(200)
)


# --- limpieza de página ---------------------------------------------------

def test_clean_page_quita_cabeceras_pies_e_indice():
    crudo = "\n".join([
        "BOLETÍN OFICIAL DEL ESTADO",
        "  Artículo 34. Jornada.  ",
        "",
        "Índice general . . . . . . 12",
        "Página 7",
        "El tiempo de trabajo se computará.",
    ])
    assert clean_page(crudo).splitlines() == [
        "Artículo 34. Jornada.",
        "El tiempo de trabajo se computará.",
    ]


# --- detección de encabezados ---------------------------------------------

def test_encabezado_reconoce_articulos_y_disposiciones():
    assert _SECTION_HEAD.match("Artículo 34. Jornada.")
    assert _SECTION_HEAD.match("Artículo único.")
    assert _SECTION_HEAD.match("Artículo 8 bis. Registro.")
    assert _SECTION_HEAD.match("Disposición adicional primera")


def test_una_referencia_interna_no_abre_seccion():
    """La razón de ser del `(?!\\d)` del regex: "artículo 83.2" es una cita
    dentro de una frase, no un encabezado. Sin esto el documento se parte por
    la mitad de un párrafo y el fragmento resultante cita el artículo que no es.
    """
    assert not _SECTION_HEAD.match("Artículo 83.2 de esta ley")
    assert not _SECTION_HEAD.match("conforme a lo previsto en el artículo 83.2")


# --- troceado de textos largos --------------------------------------------

def test_split_long_no_parte_lo_que_ya_cabe():
    texto = "a" * (MAX_CHARS - 1)
    assert _split_long(texto) == [(texto, 0)]


def test_split_long_respeta_el_tope_y_no_devuelve_trozos_vacios():
    partes = _split_long(RELLENO)
    assert len(partes) > 1
    assert all(trozo.strip() for trozo, _ in partes)
    assert all(len(trozo) <= MAX_CHARS for trozo, _ in partes)


def test_split_long_solapa_los_trozos():
    """Sin solape, un dato que cae justo en el corte se pierde para la búsqueda."""
    partes = _split_long(RELLENO)
    fin_del_primero = partes[0][1] + len(partes[0][0])
    inicio_del_segundo = partes[1][1]
    assert inicio_del_segundo < fin_del_primero


def test_split_long_no_pierde_el_final_del_texto():
    partes = _split_long(RELLENO)
    assert RELLENO.rstrip().endswith(partes[-1][0][-40:])


def test_split_long_avanza_siempre():
    """Los offsets crecen: si no, el bucle del troceado no terminaría."""
    offsets = [off for _, off in _split_long(RELLENO)]
    assert offsets == sorted(offsets)
    assert len(set(offsets)) == len(offsets)


# --- secciones y páginas ---------------------------------------------------

def test_split_sections_parte_por_articulo():
    paginas = [
        "Artículo 1. Objeto.\nEsta ley regula algo.",
        "Sigue el artículo 1 en la página dos.\nArtículo 2. Ámbito.\nSe aplica a todos.",
    ]
    assert [cabecera for cabecera, _ in _split_sections(paginas)] == [
        "Artículo 1. Objeto.",
        "Artículo 2. Ámbito.",
    ]


def test_la_pagina_se_guarda_linea_a_linea_no_por_seccion():
    """Un artículo largo ocupa varias páginas y la cita tiene que apuntar a la
    del fragmento concreto, no a la primera del artículo."""
    paginas = [
        "Artículo 1. Objeto.\nEsta ley regula algo.",
        "Sigue el artículo 1 en la página dos.",
    ]
    _, lineas = _split_sections(paginas)[0]
    assert [pagina for _, pagina in lineas] == [1, 2]

    texto, pagina_de_caracter = _page_map(lineas)
    assert texto.startswith("Esta ley regula algo.")
    assert pagina_de_caracter[0] == 1
    assert pagina_de_caracter[len("Esta ley regula algo.")] == 1  # el salto
    assert pagina_de_caracter[len("Esta ley regula algo.") + 1] == 2


# --- cita ------------------------------------------------------------------

def test_cite_nombra_documento_articulo_y_pagina():
    trozo = Chunk(
        id="et:0001", doc_id="et", doc_title="Estatuto de los Trabajadores",
        section="Artículo 35", page=12, part=1, n_parts=1, text="...",
    )
    assert trozo.cite() == "Estatuto de los Trabajadores — Artículo 35 (pág. 12)"
    assert trozo.as_dict()["page"] == 12
