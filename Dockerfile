# Image de l'Assistant IA TGR.
#
# Ne contient QUE l'application : le modèle de langage tourne dans un service
# séparé (voir docker-compose.yml). Cela permet de redéployer l'assistant sans
# retélécharger les 2 Go du modèle.

FROM python:3.11-slim

# Sorties non tamponnées : les journaux apparaissent en direct dans docker logs
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # le modèle d'embeddings est mis en cache dans un volume, pas retéléchargé
    HF_HOME=/cache/huggingface

WORKDIR /app

# Les dépendances d'abord : cette couche est réutilisée tant que
# requirements.txt ne change pas, ce qui rend les reconstructions rapides.
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install -r requirements.txt

COPY src/ ./src/
COPY static/ ./static/
COPY pyproject.toml ./

# Compte sans privilèges : un service exposé ne tourne pas en root.
RUN useradd --create-home --uid 1000 tgr \
    && mkdir -p /cache/huggingface /app/data \
    && chown -R tgr:tgr /app /cache
USER tgr

EXPOSE 8000

# Le service est prêt quand /health répond ; docker-compose attend ce signal.
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5)"

CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
