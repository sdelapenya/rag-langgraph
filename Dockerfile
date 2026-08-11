# Imagen de la demo. No sustituye al servicio de producción del servidor
# (deploy/rag-demo.service, gunicorn bajo systemd): es la forma de levantar
# esto en cualquier máquina sin instalar Python ni bajarse dependencias.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    RAG_INDEX_DIR=/app/index \
    RAG_CACHE_DIR=/cache \
    PORT=5050

WORKDIR /app

# Las dependencias van antes que el código: así tocar un .py no invalida la
# capa de pip, que es la que tarda (onnxruntime, que arrastra fastembed,
# pesa unos 200 MB).
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY *.py ./

# Sin privilegios. El proceso solo necesita leer el código y el índice, y
# escribir la caché de modelos de fastembed.
RUN useradd --create-home --uid 10001 rag \
    && mkdir -p /cache /app/index \
    && chown -R rag:rag /cache /app
USER rag

EXPOSE 5050

# start-period largo: en el primer arranque fastembed se baja el modelo de
# embeddings (unos cientos de MB) antes de que la web responda.
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=3 \
    CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:5050/salud', timeout=4)"

# -w 1 a propósito: el índice vive en memoria del proceso (~1 GB con e5) y
# más workers lo duplicarían sin ganar nada con este tráfico.
CMD ["gunicorn", "-w", "1", "-t", "120", "-b", "0.0.0.0:5050", "app:app"]
