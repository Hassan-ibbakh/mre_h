#!/usr/bin/env python
from pathlib import Path
from chromadb import PersistentClient
from langchain_huggingface import HuggingFaceEmbeddings
import torch
from config import get_settings
from ingest import ingest_document
from rag_engine import get_rag_opportunities

settings = get_settings()

# Initialiser ChromaDB et embeddings
client = PersistentClient(path=settings.CHROMA_PERSIST_DIR)
collection = client.get_or_create_collection(
    name=settings.CHROMA_COLLECTION,
    metadata={"hnsw:space": "cosine"}
)

# Déterminer le device
device = "cuda" if torch.cuda.is_available() else "cpu"
embeddings = HuggingFaceEmbeddings(
    model_name=settings.EMBEDDING_MODEL,
    model_kwargs={"device": device},
    encode_kwargs={"batch_size": settings.EMBEDDING_BATCH_SIZE}
)

# Chemin vers les fichiers invest
invest_dir = Path('data/investissement')
invest_files = sorted(invest_dir.glob('invest*.md'))

print(f"\n📁 Ingestion des fichiers invest depuis {invest_dir}...\n")

for file_path in invest_files:
    print(f"  ⬆️  {file_path.name}...")
    chunks_count = ingest_document(
        path=file_path,
        collection=collection,
        embeddings=embeddings,
        sector="investissement",
        skip_existing=False  # Force re-ingest for update
    )
    print(f"     ✓ {chunks_count} chunks indexés\n")

print("\n" + "="*60)
print("🔍 Validation des opportunités invest...\n")

# Déclenche la validation LLM
opportunities = get_rag_opportunities(settings, limit=100)

print(f"\n✓ Opportunités trouvées : {len(opportunities)}\n")
for o in opportunities:
    print(f"  📌 {o['title'][:70]}")
    print(f"     Secteur: {o['sector']} | Région: {o.get('region', 'N/A')}")
    print(f"     Source: {o['source']}\n")
