# ===== Sanad AI / MRE AI — Image de déploiement =====
# Utilisable sur : Hugging Face Spaces (Docker SDK), Render, Railway...
# Port d'écoute par défaut : 7860 (Hugging Face). Surchargé via $PORT sur Render.

FROM python:3.11-slim

# Dépendances système minimales (bcrypt, chromadb/hnswlib natifs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# torch CPU-only AVANT les autres dépendances (image ~400 Mo au lieu de 2,5 Go)
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.13.0

# Dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code applicatif + base vectorielle pré-construite
COPY . .

ENV PYTHONUNBUFFERED=1
ENV PORT=7860

EXPOSE 7860

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT} --workers 1"]
