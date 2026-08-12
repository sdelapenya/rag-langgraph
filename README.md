# RAG sobre documentos — preguntas en lenguaje natural con citas verificables

[![CI](https://github.com/sdelapenya/rag-langgraph/actions/workflows/ci.yml/badge.svg)](https://github.com/sdelapenya/rag-langgraph/actions/workflows/ci.yml)

Sistema de preguntas y respuestas sobre documentos propios: recupera los pasajes
relevantes con embeddings, responde **solo** con lo que dicen esos pasajes y
cita documento, artículo y página para que cualquiera pueda comprobarlo.

La demo funciona sobre dos textos públicos del BOE — el **Estatuto de los
Trabajadores** y la **Ley 10/2021 de trabajo a distancia** — precisamente para
que las respuestas se puedan verificar contra la fuente original.

**Demo en vivo: <https://rag.sdelapenya.dev>**

> No es asesoramiento legal: es una demo técnica sobre documentos públicos.

En este repositorio hay dos cosas, y la segunda se apoya en la primera:

| | Qué es | Dónde |
|---|---|---|
| **El RAG** | recuperación, citas y evaluación, escrito a mano y sin framework | este directorio |
| **El mismo RAG como grafo** | reexpresado con **LangGraph**: la abstención deja de ser una regla del prompt y pasa a ser un nodo con una arista condicional | [grafo/](grafo/) |

## Qué tiene de particular

Cualquiera junta un PDF y un LLM en veinte líneas. Lo que decide si sirve para
algo es lo otro:

1. **Troceado por estructura, no por longitud.** Se parte por artículos, no cada
   N caracteres, y cada fragmento arrastra su artículo y su página. Sin eso no
   hay cita fiable.
2. **Se recupera fino y se responde con el artículo entero.** El trozo que mejor
   casa con la pregunta no suele ser el que lleva el dato. Este cambio solo, sin
   tocar la búsqueda ni el modelo, subió el acierto de las respuestas **del 47 %
   al 84 %** (medido con mpnet + gpt-oss-120b, antes y después).
3. **Reescritura de la pregunta.** Nadie pregunta «compensación de gastos por
   trabajo a distancia»; pregunta «¿quién me paga la luz?». Un modelo pequeño
   traduce la pregunta al registro del documento antes de buscar.
4. **Se niega a inventar.** Ante preguntas cuya respuesta no está en el corpus,
   contesta que no la encuentra: 4 de 4 en la evaluación.
5. **Está medido.** Hay un conjunto de evaluación y un script que da los
   números; las decisiones de abajo salen de ahí, no de la intuición.

## Cómo funciona

```
pregunta del usuario
   │
   ├─► reescritura al registro legal        (llama-3.1-8b · ~200 ms)
   │
   ├─► búsqueda semántica sobre 575 fragmentos
   │      embeddings multilingual-e5-large, ONNX en CPU (~95 ms)
   │      fusión RRF de pregunta original + reescritura
   │      máximo 1 fragmento por artículo → 5 artículos distintos
   │
   ├─► expansión: de cada acierto se recupera el ARTÍCULO COMPLETO
   │      (ventana centrada en el trozo que casó, tope 4.000 caracteres)
   │
   └─► generación con las citas   (gpt-oss-20b vía Groq · ~0,7 s)
```

Sin base de datos vectorial: 575 vectores son una matriz de NumPy de 2 MB y la
búsqueda es un producto matricial. Meter Qdrant o pgvector aquí sería
infraestructura que mantener a cambio de nada; a partir de ~100.000 fragmentos
la respuesta cambia.

## Resultados de la evaluación

24 preguntas escritas como las haría una persona (20 con respuesta en el corpus,
4 cuya respuesta **no** está, para ver si se lo inventa). `k=3`.

**Recuperación** — ¿aparece el artículo correcto entre los recuperados?

| Búsqueda | Sin reescritura | Con reescritura |
|---|---|---|
| BM25 (palabras) | 0,45 | 0,65 |
| MiniLM-L12 (384 dim, 220 MB) | 0,60 | 0,60 |
| mpnet-base (768 dim, 1 GB) | 0,70 | 0,70 |
| **e5-large (1024 dim, 2,2 GB)** | 0,70 | **0,90** (MRR 0,74) |
| e5-large híbrido (BM25+vectores) | 0,60 | 0,75 |

Toda la tabla es de la misma tanda (11/08, `k=3`, 20 preguntas con respuesta en
el corpus). La versión anterior de este README daba aquí números de `k=5` sobre
19 preguntas mezclados con una configuración de `k=3`; ya no.

**La reescritura solo ayuda al modelo bueno**: sube e5-large 20 puntos
(0,70 → 0,90) y BM25 otros 20, pero a MiniLM y a mpnet no les mueve el recall.
Traducir la pregunta al registro del documento sirve si el espacio de embeddings
distingue esos matices; si no, da igual cómo se pregunte.

**Respuesta** — configuración que se publica (e5-large + reescritura + artículo
completo + gpt-oss-20b), fichero [eval-resultados-k3.json](eval-resultados-k3.json):

| Métrica | Valor |
|---|---|
| recall@3 | 0,90 |
| MRR | 0,74 |
| Respuestas con el dato correcto | 0,85 |
| Abstenciones correctas (preguntas fuera del corpus) | 4/4 |

**Bajar de `k=5` a `k=3` no costó recuperación**: recall@5 0,895 sobre las 19
preguntas de aquella tanda, recall@3 0,90 sobre las 20 de ahora, mismo MRR. El
artículo correcto, cuando se encuentra, está siempre entre los tres primeros —
los otros dos fragmentos solo engordaban el contexto.

> **Este README dijo 0,79 hasta el 11/08, y era un número mal copiado.** El 0,79
> (0,789) sale de [eval-resultados-k5-historico.json](eval-resultados-k5-historico.json), que es la tanda
> con **`k=5` y 19 preguntas**; el fichero que esta tabla cita, con la
> configuración que de verdad se publica, dice 0,70 desde el principio. No fue
> el modelo bailando: fue la celda de una tabla tomada de la ejecución
> equivocada, y la explicación que había aquí —que el generador es no
> determinista— tapaba el error en vez de encontrarlo.
>
> Ahora baila menos: el generador está a **temperatura 0** (antes 0,1) y la
> tanda del 11/08 reproduce exactamente el fichero publicado —
> acierto 0,70, recall@3 0,90, MRR 0,742. «Menos», no «nada»: Groq sirve
> `gpt-oss-20b` en lotes y ni a temperatura 0 devuelve el mismo texto siempre,
> así que una pregunta de 20 —5 puntos— puede cambiar de tanda a tanda.
>
**El acierto fue 0,70 hasta el 12/08 y lo subió el prompt, no el modelo.** Las
tres primeras reglas del `SYSTEM` eran prohibiciones seguidas («si no está, di
que no lo encuentras», dos veces más) y un modelo de 20B las sobreaplicaba:
abstenía con el dato delante. Reescrito como procedimiento —recorre los
fragmentos, y solo después ríndete—, y permitiendo responder a la parte que sí
está en vez de tirar la respuesta entera, el acierto pasa de **0,70 a 0,85 con el
mismo modelo, el mismo índice y la misma recuperación**, y las abstenciones
siguen en 4/4. De los 6 fallos arregla 3:

| Fallo | Antes | Después |
|---|---|---|
| `teletrabajo-regular` | «No encuentro esa información» con el dato delante | responde |
| `teletrabajo-gastos` | daba el dato bueno *y encima* soltaba la frase de abstención por la parte que el texto no cubre, lo que anulaba la respuesta entera | responde la parte que sí está y dice de cuál no habla |
| `iva-acotado` | «el tipo del IVA es 0 %», sin más | «0 % **para los bienes necesarios para combatir los efectos del COVID-19**» |

Con matices, que importan más que el número:

- **El prompt se escribió mirando estos mismos fallos**, así que el 0,85 está
  inflado por construcción. Para comprobarlo hay un segundo conjunto,
  [eval-preguntas-holdout.json](eval-preguntas-holdout.json): **12 preguntas
  escritas después**, sobre artículos que no participaron en el diagnóstico… y
  **no discriminan**. El prompt viejo ya saca 10/10 en ellas
  ([eval-resultados-holdout.json](eval-resultados-holdout.json)), así que solo
  permiten decir «no empeora», no «generaliza». Se publica el conjunto igualmente:
  un *held-out* que sale plano es un resultado, no un borrador.
  Lo que sí prueban es que soltar la mano en la abstención no hizo inventar nada
  — 2 trampas nuevas de 2, **6 de 6** contando las de siempre. Ese era el riesgo
  real de tocar el prompt.
- **`iva-acotado` queda a medias.** Ahora acota el supuesto, pero sigue sin decir
  hasta cuándo estuvo vigente (31/10/2020), que es lo que pide la regla.

Los 3 fallos que quedan: 2 son de recuperación (`teletrabajo-volver`,
`teletrabajo-fichar`) y el tercero, `despido-objetivo`, es de ventana — el
artículo 53 ocupa 5.481 caracteres y el recorte de 2.500 se centra en el trozo
que casó, dejando fuera «veinte días de salario por año de servicio». Mandar
además la cabecera del artículo lo arregla y sube el acierto a 0,90, pero cuesta
un **20 % más de contexto en todas las consultas** para ganar **1 pregunta de
18**, y aquí el cuello de botella es la cuota: por eso no está aplicado.

Con `llama-3.3-70b` los casos del generador salían bien en una prueba a mano
—**tanda no conservada, así que ese número no está en el repo y no lo doy**—,
pero gasta la cuota diaria de la cuenta mucho antes, así que la demo pública va
con el modelo pequeño. Es el intercambio real: acierto a cambio de que la demo
siga en pie por la tarde.

Reproducible:

```bash
.venv/bin/python3 evaluate.py --solo-recuperacion --comparar   # sin coste de API
.venv/bin/python3 evaluate.py --model e5 --mode semantic --expandir -v

# ¿le llega la respuesta al generador? Sin LLM, determinista y gratis:
RAG_MODEL=e5 .venv/bin/python3 tools/contexto_contiene.py
```

Ese último separa dos fallos que el acierto mezcla —que el sistema no le mande la
respuesta al modelo, y que se la mande y aun así falle— y es lo que decidió no
aplicar la cabecera del artículo:

```
sin cabecera     | oficial: 17/18 llegan, contexto +0 %  (falla despido-objetivo)
cabecera entera  | oficial: 18/18 llegan, contexto +25 %
```

### Lo que salió al revés

- **El híbrido BM25 + vectores empeora aquí** (0,75 frente a 0,90), y por eso va
  desactivado. RRF premia que las dos listas coincidan; cuando una de ellas es
  bastante peor que la otra, la arrastra. Barrí el peso de BM25 y el óptimo
  medido era **cero**. En un corpus con más referencias literales (códigos de
  pieza, siglas, números de expediente) esperaría lo contrario.
- **La métrica de recuperación se puede quedar tan contenta con el sistema
  roto.** Con «un fragmento por artículo» el recall subía y las respuestas
  empeoraban: el sistema recuperaba el artículo 35 pero se quedaba con la parte
  que no dice «ochenta al año». Solo se vio al medir también la respuesta.
- **El modelo grande no siempre gana.** mpnet ocupa 4 veces más que MiniLM y da
  el mismo recall. Solo e5-large compensa su tamaño.
- **Cuidado con la memoria.** Con los valores por defecto de fastembed, indexar
  llegó a ocupar 8,5 GB de RAM; con lotes de 16 y sin multiproceso baja a 2,1 GB.
- **El umbral de similitud no salva de las respuestas de más.** La idea habitual
  —«si la mejor similitud baja de X, contesta que no lo sabes»— aquí no se
  sostiene: las preguntas del corpus puntúan 0,83-0,90 y las de fuera (IVA,
  jubilación, paro) 0,83-0,86. Se solapan. Solo lo absurdo («¿cómo se hace una
  tortilla?», 0,75) queda por debajo. Lo que sí funciona contra las respuestas
  inventadas es la instrucción explícita de abstenerse, medida con preguntas
  trampa: 4 de 4.
- **Un caso a medio cerrar.** A «¿cuál es el tipo del IVA?» respondía «0 %», y no
  era una alucinación: la Ley 10/2021 lleva una disposición con IVA cero para
  material COVID hasta el 31/10/2020. Recuperaba bien y contestaba de más, sin
  acotar el supuesto. Una primera regla en el prompt no lo arregló; **sí lo hace
  la del 12/08**, que en vez de pedir «acota el supuesto» nombra el mecanismo
  —«si el fragmento es una disposición adicional o transitoria, di para qué
  supuesto se aprobó, desde cuándo y hasta cuándo»—: ahora contesta «0 % para los
  bienes necesarios para combatir los efectos del COVID-19». **Sigue sin dar la
  fecha de caducidad**, así que el caso `iva-acotado` se queda en el conjunto de
  evaluación. La lección: al modelo pequeño hay que decirle qué mirar, no lo que
  no debe hacer.

## Qué hay en el repo

| | Qué es |
|---|---|
| `rag.py`, `store.py`, `chunking.py`, `ingest.py` | el sistema: troceado, índice, búsqueda y generación |
| `app.py`, `deploy/` | la demo web (Flask + gunicorn) y su unidad de systemd |
| `grafo/` | el mismo RAG como grafo de LangGraph, con su propio README y su evaluación |
| `tools/contexto_contiene.py` | diagnóstico **sin LLM**: ¿le llega la respuesta al generador? |
| `tests/` | 38 pruebas, sin red y sin claves |
| `eval-preguntas.json` | las 24 preguntas de evaluación (20 con respuesta + 4 trampa) |
| `eval-preguntas-holdout.json` | las 12 escritas después, para validar fuera de muestra |
| `eval-reescrituras*.json` | caché de las reescrituras de consulta: evita repetir llamadas y quita ruido al comparar |
| `eval-resultados-k3.json` | **los números que publica este README** (`k=3`, 24 preguntas) |
| `eval-resultados-holdout.json` | los del conjunto de validación |
| `eval-resultados-k5-historico.json` | tanda vieja de `k=5` con 19 preguntas. **No es la configuración publicada**: se conserva porque es el origen del «0,79» que este README dio por bueno hasta el 11/08 |

Los JSON de resultados guardan, por pregunta, qué modelo la respondió. No es
decorativo: una tanda entera llegó a salir 10/10 contestada por el modelo de
respaldo, con la cuota del principal agotada y sin que nada lo dijera.

## Uso

Requisitos: Python 3.10+ y una API key de Groq, en la variable `GROQ_API_KEY` o
en un fichero de claves fuera del repositorio (`RAG_ENV_FILE=/ruta/claves.env`,
con una línea `GROQ_API_KEY=...`).

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

.venv/bin/python3 ingest.py --model e5        # construye el índice (~25 min en CPU)
.venv/bin/python3 rag.py "¿cuántas horas extra puedo hacer al año?"
.venv/bin/python3 rag.py                      # modo interactivo
```

Sobre otros documentos, sin tocar el corpus de la demo:

```bash
.venv/bin/python3 rag.py --pdf contrato.pdf "¿cuál es el plazo de entrega?"
```

Web:

```bash
RAG_MODEL=e5 .venv/bin/python3 app.py         # http://localhost:5050
```

### Con Docker

Sin instalar Python ni bajarse dependencias a mano:

```bash
python3 ingest.py              # crea ./index, que se monta como volumen
cp .env.example .env           # y pon dentro tu GROQ_API_KEY
docker compose up --build      # http://127.0.0.1:8016
```

El índice no va dentro de la imagen a propósito: pesa, caduca cada vez que
cambia el corpus y obligaría a reconstruir la imagen para reindexar. Se genera
fuera y se monta de solo lectura. La caché de modelos de embeddings vive en un
volumen con nombre para no volver a descargarla en cada arranque.

El contenedor escucha en el **8016** y no en el 8006 porque en el servidor ese
puerto lo ocupa el mismo servicio bajo systemd, y los dos tienen que poder
convivir. El proceso corre sin privilegios (uid 10001).

Despliegue en producción con systemd y túnel de Cloudflare:
[deploy/README.md](deploy/README.md).

### Ajustes por variable de entorno

| Variable | Por defecto | Para qué |
|---|---|---|
| `RAG_MODEL` | `minilm` | modelo de embeddings (`minilm`, `mpnet`, `e5`) |
| `RAG_MODE` | `semantic` | `semantic`, `keyword` o `hybrid` |
| `RAG_TOP_K` | `3` | fragmentos recuperados |
| `RAG_MAX_CHARS` | `2500` | tope del artículo que se manda al modelo |
| `RAG_LLM` | `openai/gpt-oss-20b` | modelo que redacta la respuesta |
| `RAG_LLM_QUERY` | `llama-3.1-8b-instant` | modelo que reescribe la pregunta |
| `RAG_LLM_FALLBACK` | `gemini-3.5-flash-lite` | respaldo si Groq agota la cuota |
| `RAG_CHUNK` / `RAG_OVERLAP` | `1200` / `150` | tamaño y solape del troceado |
| `RAG_CACHE_DIR` | `~/.cache/fastembed` | dónde se guardan los modelos ONNX |
| `RAG_LIMITE_IP` | `10` | preguntas por hora y por IP en la web |

## Desarrollo

```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest      # 38 tests, ~0,3 s
.venv/bin/ruff check .
```

Los tests **no llaman a ninguna API ni descargan modelos**: el índice de prueba
se construye a mano con vectores inventados, y las llamadas a Groq y Gemini se
sustituyen por dobles. Si alguna vez hace falta una clave para que pasen, es que
se ha colado una llamada de verdad.

Lo que cubren no es el porcentaje de líneas, son las decisiones que costaron
trabajo y que un cambio inocente rompería sin avisar:

- que `artículo 83.2` dentro de una frase **no** se tome por un encabezado y
  parta el documento por la mitad;
- que la página se guarde línea a línea, para que un artículo de cuatro páginas
  cite la correcta;
- que al recomponer el artículo no se duplique el texto solapado;
- que si el artículo no cabe entero, la ventana se centre en el fragmento que
  casó con la pregunta y no corte justo por donde estaba la respuesta;
- que sin fuentes el sistema se abstenga **sin gastar una llamada**;
- que una cuota diaria agotada pase a Gemini en vez de reintentar en balde.

En cada `push` y cada pull request, [CI](.github/workflows/ci.yml) pasa el
linter y los tests, y comprueba que la imagen de Docker construye.

## Estructura

| Fichero | Qué hace |
|---|---|
| [chunking.py](chunking.py) | PDF → fragmentos con artículo y página |
| [store.py](store.py) | índice, embeddings, BM25, fusión RRF, expansión a artículo |
| [ingest.py](ingest.py) | construye y guarda el índice |
| [rag.py](rag.py) | reescritura, contexto, generación y CLI |
| [app.py](app.py) | web y API JSON, con caché y límite por IP |
| [evaluate.py](evaluate.py) | métricas |
| [eval-preguntas.json](eval-preguntas.json) | conjunto de evaluación |
| [tests/](tests/) | 38 tests sin red ni claves |
| [Dockerfile](Dockerfile) · [docker-compose.yml](docker-compose.yml) | imagen sin privilegios, índice como volumen |
| [.github/workflows/ci.yml](.github/workflows/ci.yml) | linter, tests y build de la imagen |
| [grafo/](grafo/) | el mismo RAG como grafo de estados con LangGraph, con su propia evaluación |

## Límites conocidos

- **La evaluación son 23 preguntas.** Suficiente para elegir entre
  configuraciones, no para presumir de precisión. Cada punto porcentual son
  0,2 preguntas.
- **Las preguntas y el «acierto» los escribí yo**, comprobando cada dato contra
  el texto. Un conjunto ciego, hecho por otra persona, sería más honesto.
- El corpus son 2 documentos y 575 fragmentos. La búsqueda por fuerza bruta deja
  de valer bastante más arriba, pero no es gratis para siempre.
- Los PDFs del BOE tienen texto extraíble. Con PDFs escaneados haría falta OCR,
  que no está.
- La demo pública consume API en cada pregunta nueva. **El techo no es el precio,
  es la cuota.** El plan gratuito de Groq da 8.000 tokens/minuto y **200.000 al
  día** con `gpt-oss-20b`; a ~2.100 tokens por pregunta salen unas **95 preguntas
  diarias**. Contramedidas, por orden de lo que aporta cada una:

  | Medida | Efecto |
  |---|---|
  | `k=3` y artículo de 2.500 caracteres | **−54 % de contexto** (3.650 → 1.660 tokens) |
  | Respaldo en Gemini al agotar la cuota | +500 preguntas/día (su plan gratis cuenta 1.000 **peticiones**, no tokens, y cada pregunta gasta 2: reescritura y respuesta) |
  | Caché LRU de respuestas | las preguntas de ejemplo, que son las que más se pulsan, no llegan a la API |
  | Límite de 10 preguntas/hora por IP | que un visitante no funda la cuota del día |

  Con tráfico de verdad tocaría pagar Groq, que son unos **0,20 $ por cada 1.000
  preguntas** — otra vez: el problema es el tope, no el precio.

  > ⚠️ El plan gratuito de Gemini **entrena con lo que se le envía**. Aquí da
  > igual (el corpus es BOE, texto público). Para documentos de cliente, ni el
  > respaldo ni nada: plan de pago con el tratamiento de datos por escrito.
