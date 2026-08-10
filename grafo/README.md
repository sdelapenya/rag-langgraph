# RAG como grafo (LangGraph)

El RAG de [la carpeta de arriba](..) —recuperación híbrida sobre normativa laboral
española, escrito a mano y publicado en `rag.sdelapenya.dev`— reexpresado como
un grafo de estados con **LangGraph 1.2**, para hacer explícita la decisión que
allí estaba escondida dentro de un prompt: **cuándo NO hay que responder**.

No es un tutorial ni un clon: importa el código del RAG original tal cual (misma
recuperación, mismo índice, mismo conjunto de evaluación) y solo cambia la forma
en que se toman las decisiones.

## El grafo

```mermaid
graph TD;
    __start__([inicio]) --> recuperar
    recuperar --> evaluar
    evaluar -. "el contexto responde" .-> generar
    evaluar -. "no responde" .-> abstenerse
    generar --> __end__([fin])
    abstenerse --> __end__
```

| Nodo | Qué hace | Coste |
|------|----------|-------|
| `recuperar` | reescribe la pregunta al registro del documento y busca en el índice (k=3) | 1 llamada a un modelo pequeño |
| `evaluar` | ¿estos fragmentos contienen la respuesta? Suelo de similitud + modelo juez | 1 llamada a un modelo pequeño |
| `generar` | redacta la respuesta citando `[n]` | 1 llamada al modelo grande |
| `abstenerse` | cierra con «No encuentro esa información en los documentos.» y el motivo | 0 |

La arista condicional sale de `evaluar`. Ese es el cambio de fondo:

- **En el RAG original**, la abstención es la regla nº 1 del prompt de
  generación. Al modelo grande le llegan los fragmentos siempre, incluso cuando
  no valen, y unas veces se abstiene y otras no. El motivo no queda registrado
  en ninguna parte.
- **Aquí**, la decisión es un nodo con un modelo pequeño, ocurre *antes* de
  generar, y el porqué se queda escrito en el estado (`motivo`) — se puede
  enseñar, registrar y auditar.

## Cómo se usa

```bash
python3 grafo.py "¿cuántas horas extra puedo hacer al año?" -v
python3 grafo.py "¿cuál es el tipo general del IVA?" -v      # se abstiene
python3 grafo.py --mermaid                                   # el grafo, sin cargar el índice
python3 evaluar_grafo.py                                     # las 24 preguntas de evaluación
python3 calibrar_umbral.py                                   # de dónde sale el umbral
```

Con `-v` se ve la ruta que siguió la pregunta por el grafo, la reescritura, las
similitudes y qué modelo respondió:

```
  ruta        recuperar -> evaluar -> abstenerse
  reescritura ¿Cuál es la tasa general del Impuesto sobre el Valor Añadido en España?
  similitudes [0.8353, 0.8158, 0.8089]  (umbral 0.8)
  modelos     juez=llama-3.1-8b-instant  respuesta=-
  ahorrado    una llamada a openai/gpt-oss-20b
```

## Resultados

Las mismas 24 preguntas del RAG original (20 con respuesta en el corpus, 4 de
materias que no regula), misma configuración: e5, semantic, k=3.

| | RAG original | Este grafo |
|--|--|--|
| acierto de respuesta | 0,70 | 0,75 |
| abstención correcta (4 preguntas de fuera) | 4/4 | **4/4** |
| llamadas al modelo grande | 24/24 | **20/24** |
| abstenciones provocadas por el juez sobre preguntas buenas | — | **0** |

Lo que hay que leer en esa tabla:

- **Las 4 llamadas ahorradas son las 4 preguntas sin respuesta.** El grafo las
  corta en `evaluar`, con un modelo pequeño, en vez de mandarle al grande un
  contexto que no sirve. Con un corpus mayor y usuarios reales, esa proporción
  es lo que se factura.
- **Cero abstenciones indebidas**: el juez no cortó ni una sola de las 20
  preguntas que sí tenían respuesta. Era el riesgo de meter un filtro delante y
  es la columna que más importa.
- **La diferencia de acierto (0,70 → 0,75) no cuenta como mejora.** El grafo no
  toca la generación: son las mismas fuentes y el mismo prompt. Es variación
  entre ejecuciones de un modelo no determinista sobre 20 preguntas — una
  pregunta arriba o abajo son 5 puntos.
- Las 4 preguntas que aún acaban en «no lo encuentro» (`despido-objetivo`,
  `teletrabajo-regular`, `teletrabajo-volver`, `teletrabajo-fichar`) las corta
  **el modelo grande al redactar**, no el juez: es el comportamiento que ya
  tenía el RAG original. Por eso la evaluación separa `cortadas_por_el_juez` de
  `abstenidas_al_generar`; sumarlas escondería de quién es la culpa.

Recall@3 (0,89) y MRR (0,74) no se recalculan: la recuperación es literalmente
la misma función, así que son los números de `evaluate.py` sin tocar.

### El juez, en su segunda versión

La primera versión del nodo `evaluar` **tumbó 5 preguntas buenas de 20**, todas
por el mismo motivo: pedía la palabra literal. Decía *«el texto no menciona el
fichaje»* cuando la ley dice «registro horario», o *«no menciona un plazo en
días»* cuando el artículo pide «el treinta por ciento de la jornada en tres
meses».

Es el mismo desajuste que el RAG ya resuelve al buscar —lenguaje corriente
contra lenguaje legal— y la solución estaba en el estado sin usar: **al juez
ahora se le pasa también la reescritura formal de la pregunta**, que ya se
calculó en `recuperar` y no cuesta nada. Con eso, y con un prompt que exige
literalidad solo donde debe (una cifra, un importe, un plazo que no aparece),
las 5 volvieron a pasar sin que se colara ninguna de las 4 de fuera.

## Decisiones

**Entorno virtual y carpeta aparte.** El RAG del directorio padre sirve un proceso en
producción (`rag-demo.service`, puerto 8006). Instalar LangGraph en su entorno
para probar algo es arriesgar la demo que sí funciona por una dependencia
transitiva. Aquí hay un `.venv` propio con las mismas versiones de las librerías
compartidas; el código y el índice del RAG se leen de su sitio, sin copiarlos ni
modificarlos. El puente son diez líneas: [`puente.py`](puente.py).

**El juez es un modelo pequeño** (`llama-3.1-8b-instant`). Decidir si un texto
contiene un dato es clasificar, no redactar. Si el filtro costara lo mismo que
la respuesta, no filtraría nada: solo añadiría latencia.

**El umbral de similitud es un suelo, no un discriminador — y eso está medido.**
La idea original era cortar por similitud antes de gastar ni la llamada al juez.
[`calibrar_umbral.py`](calibrar_umbral.py) dice que no se puede: con e5 las 20
preguntas del corpus caen entre 0,828 y 0,895 y las 4 de fuera entre 0,833 y
0,862 — **se solapan**. e5 comprime todos los cosenos en una franja estrecha, así
que no hay umbral que separe «lo que está en el corpus» de «lo que no». El
umbral se queda en 0,80, por debajo de todo lo observado, donde sí sirve: caza
preguntas ajenas al dominio (*«¿cuál es la capital de Mongolia?»* → 0,72) sin
gastar una llamada. Lo fino lo hace el juez. La salida de la calibración está en
[`calibracion-umbral.txt`](calibracion-umbral.txt).

**Ante una respuesta rara del juez, se genera.** Si el veredicto no empieza por
«NO», se sigue a `generar`. Preferimos gastar una llamada de más a callarnos ante
una pregunta que sí tenía respuesta en los documentos.

## Límites

- Es un grafo lineal con una bifurcación: no hay ciclos, ni memoria entre
  preguntas, ni checkpointer. LangGraph luce cuando hay reintentos y estado
  persistente; aquí se usa lo que el problema pide y nada más.
- No está publicado: se ejecuta por CLI. La demo web sigue siendo la del RAG
  original, en `rag.sdelapenya.dev`.
- El conjunto de evaluación son 24 preguntas. Sirve para comparar dos
  variantes del mismo sistema, no para afirmar nada en general.
- Un juez con el mismo tipo de modelo que el generador comparte sus puntos
  ciegos. Lo honesto sería medirlo contra un juez humano; con 24 preguntas se
  puede, pero aún no está hecho.

## Ficheros

| Fichero | Qué es |
|---------|--------|
| [`grafo.py`](grafo.py) | el grafo, los cuatro nodos y la CLI |
| [`puente.py`](puente.py) | hace importable el RAG del directorio padre sin copiar código |
| [`evaluar_grafo.py`](evaluar_grafo.py) | pasa las 24 preguntas por el grafo |
| [`calibrar_umbral.py`](calibrar_umbral.py) | mide si la similitud puede decidir sola |
| `eval-grafo.json` | detalle pregunta a pregunta de la última evaluación |

## Instalación

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Necesita `GROQ_API_KEY` (y opcionalmente `GEMINI_API_KEY` como respaldo) en el
entorno o en un fichero de claves (`RAG_ENV_FILE=/ruta/claves.env`), igual que
el RAG original, y que el índice del RAG (`../index/`) esté construido con
`ingest.py`. Si el RAG está en otra ruta: `RAG_DIR=/ruta/al/rag python3 grafo.py ...`
