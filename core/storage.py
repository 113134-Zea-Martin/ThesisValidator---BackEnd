import sqlite3
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path

class DocumentStorage:
    def __init__(self, db_path: str = "./data/documents.db"):
        """Inicializa almacenamiento persistente con SQLite"""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        
        # Mantener cache en memoria para compatibilidad
        self.documents: Dict[str, Dict] = {}
        self.embeddings: Dict[str, List[float]] = {}
        self._load_cache()

    def _init_database(self):
        """Crea las tablas si no existen"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL
                )
            """)
            conn.commit()

    def _load_cache(self):
        """Carga todos los documentos en memoria al inicio"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT document_id, title, content, created_at FROM documents")
            for row in cursor.fetchall():
                doc_id, title, content, created_at = row
                self.documents[doc_id] = {
                    "title": title,
                    "content": content,
                    "created_at": datetime.fromisoformat(created_at)
                }

    def save_document(self, title: str, content: str) -> str:
        """Guarda documento en SQLite y cache"""
        document_id = f"{len(self.documents) + 1}"
        created_at = datetime.utcnow()
        
        # Guardar en base de datos
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO documents (document_id, title, content, created_at) VALUES (?, ?, ?, ?)",
                (document_id, title, content, created_at.isoformat())
            )
            conn.commit()
        
        # Actualizar cache
        self.documents[document_id] = {
            "title": title,
            "content": content,
            "created_at": created_at,
        }
        return document_id
    
    def list_documents(self) -> List[Dict]:
        return [
            {"document_id": doc_id, "title": doc["title"], "content": doc["content"]}
            for doc_id, doc in self.documents.items()
        ]
    

    def get_document(self, document_id: str) -> Optional[Dict]:
        return self.documents.get(document_id)

    def create_embedding(self, document_id: str, content: str) -> None:
        # Dummy embedding: convert first 100 chars to floats
        embedding = [float(ord(c)) / 100.0 for c in content[:100]]
        self.embeddings[document_id] = embedding

    def search(self, query: str) -> List[Dict]:
        results = []
        for doc_id, doc in self.documents.items():
            if query.lower() in doc["content"].lower() or query.lower() in doc["title"].lower():
                results.append({
                    "document_id": doc_id,
                    "title": doc["title"],
                    "content_snippet": doc["content"][:160],
                    "similarity_score": 0.9,
                })
        return results

    def answer_question(self, question: str) -> tuple[str, List[Dict]]:
        context = []
        for doc_id, doc in self.documents.items():
            if question.lower() in doc["content"].lower():
                context.append({
                    "document_id": doc_id,
                    "content_snippet": doc["content"][:160],
                    "similarity_score": 0.8,
                })
        answer = "Respuesta basada en coincidencias simples de texto."
        return answer, context