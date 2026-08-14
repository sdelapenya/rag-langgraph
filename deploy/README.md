# Despliegue de la demo

La demo se sirve con Gunicorn en `127.0.0.1:8006` y sale a internet por el túnel
de Cloudflare ya existente. No se abre ningún puerto en el router.

## 1. Servicio systemd

```bash
sudo cp ~/lab/rag/deploy/rag-demo.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now rag-demo
systemctl status rag-demo --no-pager
curl -s localhost:8006/salud
```

Las claves se leen de `~/secrets/rag-demo.env` (permisos 600), que contiene
**solo** `GROQ_API_KEY` y `GEMINI_API_KEY`. Un fichero de claves de uso general
no vale aquí: suele llevar credenciales de otros servicios que este proceso no
tiene por qué ver.

`GEMINI_API_KEY` es el respaldo para cuando Groq agota la cuota diaria; se saca
gratis y sin tarjeta en [ai.google.dev](https://ai.google.dev). Si no está, la
demo funciona igual: simplemente devuelve el 503 de siempre al quedarse sin Groq.

## 2. Hostname en Cloudflare

⚠️ El túnel de este servidor **se gestiona desde el dashboard de Zero Trust**, no
desde `~/.cloudflared/config.yml` — editar el fichero local no tiene ningún
efecto. Hay que darlo de alta a mano:

1. Entrar en Cloudflare Zero Trust → **Networks → Tunnels** → el túnel activo →
   **Public Hostnames** → *Add a public hostname*.
2. Subdomain `rag`, domain `sdelapenya.dev`, Service `HTTP` → `localhost:8006`.
3. Guardar y comprobar: `curl -sI https://rag.sdelapenya.dev` debe dar 200.

## 3. Comprobaciones tras desplegar

```bash
journalctl -u rag-demo -f                      # arranque y errores
curl -s localhost:8006/salud | python3 -m json.tool
```

El primer arranque carga el modelo de embeddings en memoria (unos segundos). Si
tarda mucho más, seguramente esté descargando el modelo: se cachea en
`~/.cache/fastembed` (configurable con `RAG_CACHE_DIR`).

## Coste y protección

- Cada pregunta = **2 llamadas** (reescritura con `qwen/qwen3.6-27b` +
  respuesta con `openai/gpt-oss-20b`, que es lo que fija `RAG_LLM` en el servicio)
  y **~2.100 tokens medidos** desde el 02/08 (antes 4.400: se bajó `RAG_TOP_K` a 3
  y el artículo a 2.500 caracteres, −54 % de contexto sin perder acierto).
- ⚠️ **El cuello de botella no es el precio, es la cuota.** El plan gratuito de
  Groq da 8.000 tokens/minuto y **200.000 al día**: unas **95 preguntas diarias**.
  Al agotarse, el servicio **pasa solo a Gemini** (`RAG_LLM_FALLBACK`), cuyo plan
  gratis cuenta 1.000 peticiones/día en vez de tokens — como cada pregunta gasta
  dos (reescritura y respuesta), son **~500 preguntas más al día**. En los logs se
  ve el salto (`Groq sin cuota …, pasando a …`) y la web muestra qué modelo
  respondió.
- Si aun así se queda corto, el Dev Tier de Groq son **~0,20 $ por 1.000
  preguntas** (decisión pendiente de Sergio a 2026-08-02).
- La demo limita a **10 preguntas por hora y por IP** (`RAG_LIMITE_IP`) y corta
  la pregunta a 300 caracteres. La IP real llega en la cabecera
  `CF-Connecting-IP` que pone el túnel.
- No hay subida de documentos: el corpus se fija con `ingest.py`. Para un cliente
  con sus propios PDFs, la subida iría detrás de autenticación.
