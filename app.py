#!/usr/bin/env python3
"""
Web de la demo: pregunta en lenguaje natural sobre el corpus indexado y
devuelve la respuesta con las citas (documento, artículo y página).

Variables de entorno:
  RAG_MODEL       modelo de embeddings a usar (índice ya construido)
  RAG_MODE        hybrid | semantic | keyword   (por defecto: semantic)
  RAG_TOP_K       fragmentos recuperados por pregunta
  RAG_MAX_CHARS   tope de caracteres del artículo que se manda al modelo
  RAG_LIMITE_IP   preguntas por hora y por IP (0 = sin límite)
  PORT            puerto (5050)

La demo es pública y cada pregunta cuesta una llamada de API, así que hay
límite por IP. No hay subida de documentos: el corpus se fija con ingest.py.
Si Groq agota la cuota diaria, rag.py pasa solo a Gemini (ver MODEL_FALLBACK).
"""
from __future__ import annotations

import os
import time
from collections import OrderedDict, defaultdict, deque

from flask import Flask, jsonify, render_template_string, request

from rag import MODEL_LLM, TOP_K as RAG_TOP_K, _load_env_key, answer, expand_query
from store import DEFAULT_MODE, DEFAULT_MODEL, Index

MODE = DEFAULT_MODE
TOP_K = RAG_TOP_K
EXPANDIR = os.getenv("RAG_EXPANDIR", "1") == "1"
# La cuota diaria de Groq da para unas 80 preguntas: con 20/hora bastaba un
# visitante para fundir un cuarto del día. 10 sigue sobrando para probar la demo.
LIMITE_IP = int(os.getenv("RAG_LIMITE_IP", "10"))  # preguntas/hora/IP
CACHE_MAX = int(os.getenv("RAG_CACHE_RESPUESTAS", "300"))
MAX_CHARS_PREGUNTA = 300

app = Flask(__name__)
INDEX = Index.load(DEFAULT_MODEL)
API_KEY = os.getenv("GROQ_API_KEY") or _load_env_key()
_peticiones: dict[str, deque] = defaultdict(deque)
# Las preguntas de ejemplo se repiten mucho (todo el mundo pulsa las mismas) y
# el corpus no cambia: guardar la respuesta evita repetir la llamada de API.
_cache: OrderedDict[str, dict] = OrderedDict()


def clave_cache(pregunta: str) -> str:
    return " ".join(pregunta.lower().split())


# la primera consulta carga el modelo de embeddings (unos segundos): se hace
# aquí, al arrancar, para que no la pague el primero que entre en la web
INDEX.embed_query("calentando el modelo")

EJEMPLOS = [
    "¿Cuántas horas extra puedo hacer al año?",
    "Trabajo desde casa, ¿quién paga el ordenador y la luz?",
    "Si me despiden y es improcedente, ¿cuánto me corresponde?",
    "¿Cuánto descanso me toca entre dos jornadas?",
    "¿Puedo dejar de teletrabajar y volver a la oficina?",
]


def ip_cliente() -> str:
    # detrás del túnel de Cloudflare, la IP real viene en esta cabecera
    return request.headers.get("CF-Connecting-IP") or request.remote_addr or "?"


def excede_limite(ip: str) -> bool:
    if LIMITE_IP <= 0:
        return False
    ahora = time.time()
    cola = _peticiones[ip]
    while cola and ahora - cola[0] > 3600:
        cola.popleft()
    if len(cola) >= LIMITE_IP:
        return True
    cola.append(ahora)
    return False


PAGE = """
<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Demo RAG — preguntas sobre normativa laboral</title>
<meta name="description" content="Preguntas en lenguaje natural sobre el Estatuto de los Trabajadores y la Ley 10/2021, respondidas solo con lo que dicen los documentos y citando artículo y página.">
<!-- Open Graph: sin esto, al pegar el enlace en LinkedIn o WhatsApp sale una
     tarjeta sin título ni descripción (le pasó a chat.sdelapenya.dev). La
     miniatura sigue habiendo que subirla a mano: no hay imagen que servir. -->
<meta property="og:type" content="website">
<meta property="og:site_name" content="sdelapenya.dev">
<meta property="og:title" content="RAG — respuestas con citas verificables">
<meta property="og:description" content="Preguntas en lenguaje natural sobre normativa laboral, respondidas solo con lo que dicen los documentos y citando artículo y página. Evaluado: recall@3 0,89 · acierto 0,79.">
<meta property="og:url" content="https://rag.sdelapenya.dev/">
<meta name="twitter:card" content="summary">
<style>
  :root { --tinta:#0f172a; --suave:#64748b; --linea:#e2e8f0; --fondo:#f8fafc; --acento:#1d4ed8; }
  * { margin:0; padding:0; box-sizing:border-box; }
  body { font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif;
         background:var(--fondo); color:var(--tinta); line-height:1.55; }
  .wrap { max-width:760px; margin:0 auto; padding:44px 20px 80px; }
  header h1 { font-size:23px; letter-spacing:-.2px; }
  header p { color:var(--suave); font-size:14.5px; margin-top:6px; }
  .card { background:#fff; border:1px solid var(--linea); border-radius:12px;
          padding:22px 24px; margin-top:22px; }
  textarea { width:100%; border:1px solid #cbd5e1; border-radius:9px; padding:12px;
             font:inherit; font-size:15px; min-height:74px; resize:vertical; }
  textarea:focus { outline:2px solid var(--acento); outline-offset:1px; border-color:transparent; }
  button { margin-top:12px; background:var(--tinta); color:#fff; border:0; border-radius:9px;
           padding:11px 20px; font-size:14.5px; cursor:pointer; }
  button:hover { background:#1e293b; }
  button[disabled] { opacity:.55; cursor:progress; }
  .ejemplos { display:flex; flex-wrap:wrap; gap:8px; margin-top:14px; }
  .ejemplos button { margin:0; background:#eef2ff; color:#3730a3; font-size:13px;
                     padding:6px 11px; border-radius:99px; }
  .ejemplos button:hover { background:#e0e7ff; }
  .respuesta { margin-top:20px; padding:16px 18px; background:#f0fdf4;
               border:1px solid #bbf7d0; border-radius:10px; white-space:pre-wrap; font-size:15px; }
  .respuesta.vacia { background:#fffbeb; border-color:#fde68a; }
  h2 { font-size:13px; text-transform:uppercase; letter-spacing:.6px; color:var(--suave);
       margin-bottom:10px; font-weight:600; }
  .fuente { border-top:1px solid var(--linea); padding:11px 0; font-size:14px; }
  .fuente:first-of-type { border-top:0; }
  .fuente b { color:var(--acento); }
  .fuente details summary { cursor:pointer; color:var(--suave); font-size:13px; margin-top:4px; }
  .fuente pre { white-space:pre-wrap; font:inherit; font-size:13.5px; color:#334155;
                background:var(--fondo); padding:10px; border-radius:7px; margin-top:8px; }
  .meta { color:var(--suave); font-size:12.5px; margin-top:18px; }
  .meta code { background:#eef2f6; padding:1px 5px; border-radius:4px; }
  .aviso { font-size:12.5px; color:var(--suave); margin-top:26px; border-top:1px solid var(--linea);
           padding-top:16px; }
  a { color:var(--acento); }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>Pregunta a los documentos</h1>
    <p>Demo de RAG sobre el <b>Estatuto de los Trabajadores</b> y la <b>Ley de Trabajo a Distancia</b>
       (textos consolidados del BOE). Responde solo con lo que dice el texto y cita artículo y página,
       para que puedas comprobarlo.</p>
  </header>

  <div class="card">
    <form id="form">
      <textarea id="pregunta" name="pregunta" maxlength="{{ max_chars }}"
                placeholder="Escribe tu pregunta…" required></textarea>
      <button type="submit" id="enviar">Preguntar</button>
    </form>
    <div class="ejemplos">
      {% for e in ejemplos %}<button type="button" onclick="usar(this)">{{ e }}</button>{% endfor %}
    </div>
    <div id="salida"></div>
  </div>

  <p class="meta">
    {{ meta.n_chunks }} fragmentos indexados de {{ meta.documents|length }} documentos ·
    embeddings <code>{{ meta.model.split('/')[-1] }}</code> ·
    búsqueda <code>{{ modo }}</code> · generación <code>{{ llm }}</code> vía Groq
  </p>

  <p class="aviso">
    Demo técnica con documentos públicos; <b>no es asesoramiento legal</b>. El texto puede no
    reflejar la última reforma: verifica siempre en <a href="https://www.boe.es">boe.es</a>.
    Límite de {{ limite }} preguntas por hora. Código y evaluación:
    <a href="https://github.com/sdelapenya">repositorio</a>.
  </p>
</div>

<script>
const form = document.getElementById('form');
const salida = document.getElementById('salida');
const boton = document.getElementById('enviar');

function usar(b) {
  document.getElementById('pregunta').value = b.textContent;
  form.requestSubmit();
}

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const pregunta = document.getElementById('pregunta').value.trim();
  if (!pregunta) return;
  boton.disabled = true;
  salida.innerHTML = '<div class="respuesta">Buscando en los documentos…</div>';
  try {
    const r = await fetch('/api/preguntar', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({pregunta})
    });
    const d = await r.json();
    if (d.error) {
      salida.innerHTML = '<div class="respuesta vacia">' + esc(d.error) + '</div>';
    } else {
      let html = '<div class="respuesta">' + esc(d.respuesta) + '</div>';
      html += '<div style="margin-top:22px"><h2>Fuentes citadas</h2>';
      for (const f of d.fuentes) {
        html += '<div class="fuente"><b>[' + f.n + ']</b> ' + esc(f.cita) +
                '<details><summary>ver fragmento</summary><pre>' + esc(f.texto) + '</pre></details></div>';
      }
      html += '</div><p class="meta">búsqueda ' + d.ms_busqueda + ' ms · respuesta ' +
              d.ms_respuesta + ' ms · ' + esc(d.modelo) + '</p>';
      salida.innerHTML = html;
    }
  } catch (err) {
    salida.innerHTML = '<div class="respuesta vacia">No se pudo completar la consulta.</div>';
  }
  boton.disabled = false;
});

function esc(s) {
  const d = document.createElement('div');
  d.textContent = s == null ? '' : s;
  return d.innerHTML;
}
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(
        PAGE,
        meta=INDEX.meta,
        modo=MODE,
        llm=MODEL_LLM,
        ejemplos=EJEMPLOS,
        limite=LIMITE_IP,
        max_chars=MAX_CHARS_PREGUNTA,
    )


@app.route("/api/preguntar", methods=["POST"])
def preguntar():
    datos = request.get_json(silent=True) or {}
    pregunta = (datos.get("pregunta") or "").strip()[:MAX_CHARS_PREGUNTA]
    if not pregunta:
        return jsonify({"error": "Escribe una pregunta."}), 400

    clave = clave_cache(pregunta)
    if clave in _cache:
        _cache.move_to_end(clave)
        return jsonify({**_cache[clave], "cacheada": True})

    if excede_limite(ip_cliente()):
        return jsonify({"error": f"Has llegado al límite de {LIMITE_IP} preguntas por hora."}), 429

    t0 = time.perf_counter()
    reescritura = expand_query(pregunta, API_KEY) if EXPANDIR else ""
    hits = INDEX.search(pregunta, k=TOP_K, mode=MODE, max_per_section=1,
                        extra_queries=[reescritura] if reescritura else None)
    ms_busqueda = round((time.perf_counter() - t0) * 1000)

    t1 = time.perf_counter()
    try:
        resultado = answer(INDEX, pregunta, k=TOP_K, mode=MODE, api_key=API_KEY, hits=hits)
    except Exception as err:  # cuota agotada, corte de red, modelo caído…
        app.logger.warning("fallo al generar la respuesta: %s", err)
        return jsonify({
            "error": "El servicio de generación no está disponible ahora mismo. "
                     "Vuelve a intentarlo en unos minutos."
        }), 503
    ms_respuesta = round((time.perf_counter() - t1) * 1000)

    salida = {
        "pregunta": pregunta,
        "reescritura": reescritura,
        "respuesta": resultado["answer"],
        "modelo": resultado["model"],
        "fuentes": [
            {"n": f["n"], "cita": f["cite"], "texto": f["text"]}
            for f in resultado["sources"]
        ],
        "ms_busqueda": ms_busqueda,
        "ms_respuesta": ms_respuesta,
    }
    _cache[clave] = salida
    while len(_cache) > CACHE_MAX:
        _cache.popitem(last=False)
    return jsonify(salida)


@app.route("/salud")
def salud():
    return jsonify({"ok": True, **INDEX.meta})


if __name__ == "__main__":
    if not API_KEY:
        raise SystemExit("Falta GROQ_API_KEY: ponla en el entorno o en el fichero de claves (RAG_ENV_FILE)")
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5050")))
