# RAG sobre documentos — preguntas en lenguaje natural con citas verificables

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
| BM25 (palabras) | 0,42 | 0,68 |
| MiniLM-L12 (384 dim, 220 MB) | 0,79 | 0,84 |
| mpnet-base (768 dim, 1 GB) | 0,79 | 0,84 |
| **e5-large (1024 dim, 2,2 GB)** | 0,89 | **0,89** (MRR 0,73) |
| e5-large híbrido (BM25+vectores) | 0,68 | 0,79 |

**Respuesta** — configuración que se publica (e5-large + reescritura + artículo
completo + gpt-oss-20b), fichero [eval-resultados-k3.json](eval-resultados-k3.json):

| Métrica | Valor |
|---|---|
| recall@3 | 0,89 |
| MRR | 0,74 |
| Respuestas con el dato correcto | 0,79 |
| Abstenciones correctas (preguntas fuera del corpus) | 4/4 |

**Bajar de `k=5` a `k=3` no costó recuperación**: 17 de 19 preguntas en ambos
casos, mismo MRR. El artículo correcto, cuando se encuentra, está siempre entre
los tres primeros — los otros dos fragmentos solo engordaban el contexto.

> **Por qué el acierto es 0,79 y no el 0,74 que imprime una tanda suelta.** El
> generador va a temperatura 0,1, así que no es determinista: en la tanda del
> 02/08 falló `teletrabajo-gastos`, que **al repetirla acierta 4 de 4 veces,
> tanto con `k=3` como con `k=5`**. Una pregunta sobre 19 vale 5 puntos, y esa
> pregunta no se pierde por el recorte de contexto sino por el muestreo.
> Además el criterio es tosco: aquella respuesta traía el dato correcto *y
> encima* avisaba de que el texto no dice quién paga la luz — y la frase de
> aviso se cuenta como abstención total.
>
> Para que el número deje de bailar habría que **poner la temperatura a 0** y
> volver a medir. Pendiente.

De los fallos, 2 son de recuperación y el resto del generador: encontró el
artículo correcto y aun así dijo que no lo encontraba. Con `llama-3.3-70b` esos
casos salen bien (acierto 0,84), pero gasta la cuota diaria de la cuenta mucho
antes, así que la demo pública va con el modelo pequeño. Es el intercambio real:
5 puntos de acierto a cambio de que la demo siga en pie por la tarde.

Reproducible:

```bash
.venv/bin/python3 evaluate.py --solo-recuperacion --comparar   # sin coste de API
.venv/bin/python3 evaluate.py --model e5 --mode semantic --expandir -v
```

### Lo que salió al revés

- **El híbrido BM25 + vectores empeora aquí** (0,79 frente a 0,89), y por eso va
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
- **Un caso que sigue abierto.** A «¿cuál es el tipo del IVA?» responde «0 %», y
  no es una alucinación: la Ley 10/2021 lleva una disposición con IVA cero para
  material COVID hasta el 31/10/2020. Está recuperando bien y contestando de
  más, sin acotar el supuesto. Añadir una regla al prompt no lo arregló con el
  modelo pequeño; el de 120B sí matiza («en el supuesto regulado»). Queda como
  caso `iva-acotado` en el conjunto de evaluación.

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

Despliegue con systemd y túnel de Cloudflare: [deploy/README.md](deploy/README.md).

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
