
from typing import List
import logging
from fastapi import FastAPI, HTTPException, status
from schemas.schemas import (
    DocumentUpload,
    DocumentResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    SearchQuery,
    SearchResponse,
    QuestionRequest,
    AnswerResponse,
)
from services.services import DocumentService
from core.storage import DocumentStorage

logger = logging.getLogger(__name__)

def register_routes(app: FastAPI):
    storage = DocumentStorage()
    service = DocumentService(storage)

    @app.post("/upload", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED, summary="Cargar documento")
    def upload_document(document: DocumentUpload):
        try:
            return service.upload_document(document)
        except Exception as e:
            logger.error("Error en /upload: %s", e)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error interno al cargar el documento")
        
    # Traer todos los documentos cargados
    @app.get("/documents", response_model=List[DocumentResponse], summary="Listar documentos")
    def list_documents():
        try:
            documents = service.list_documents()
            return documents
        except Exception as e:
            logger.error("Error en /documents: %s", e)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error interno al listar documentos")
        
    @app.post("/generate-embeddings", response_model=EmbeddingResponse, summary="Generar embeddings")
    def generate_embeddings(request: EmbeddingRequest):
        try:
            return service.generate_embedding(request)
        except ValueError:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Documento no encontrado")
        except Exception as e:
            logger.error("Error en /generate-embeddings: %s", e)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error interno al generar embeddings")

    # Traer todos los emmbedings generados
    @app.get("/embeddings", summary="Listar embeddings generados en ChromaDB")
    def _list_embeddings():
        try:
            # # Obtener colección de ChromaDB
            # collection = service._get_chroma_collection()
            
            # # Recuperar todos los datos indexados
            # all_data = collection.get(include=["embeddings", "documents", "metadatas"])
            
            # # Formatear respuesta
            # embeddings_info = {
            #     "total_chunks": len(all_data["ids"]),
            #     "chunks": []
            # }
            
            # for i, chunk_id in enumerate(all_data["ids"]):
            #     # ✅ Verificar que embeddings existe Y que tiene datos
            #     embedding_length = 0
            #     if all_data["embeddings"] is not None and len(all_data["embeddings"]) > i:
            #         embedding_length = len(all_data["embeddings"][i])
                
            #     embeddings_info["chunks"].append({
            #         "chunk_id": chunk_id,
            #         "document_id": all_data["metadatas"][i].get("source"),
            #         "title": all_data["metadatas"][i].get("title"),
            #         "chunk_index": all_data["metadatas"][i].get("chunk_index"),
            #         "content_snippet": all_data["documents"][i][:200],
            #         "embedding_length": embedding_length
            #     })
            
            # return embeddings_info
            return service.vector_store.list_all_embeddings()
        except Exception as e:
            logger.error("Error al listar embeddings: %s", e)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                detail="Error interno al listar embeddings"
            )

    @app.post("/search", response_model=SearchResponse, summary="Buscar documentos")
    def search_documents(query: SearchQuery):
        try:
            if _contains_sensitive_content(query.query):
                logger.warning("Consulta sensible detectada")
                return SearchResponse(results=[])
            return service.search_documents(query)
        except Exception as e:
            logger.error("Error en /search: %s", e)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error interno en la búsqueda")

    @app.post("/ask", response_model=AnswerResponse, summary="Hacer pregunta")
    def ask_question(question: QuestionRequest):
        try:
            if _contains_inappropriate_content(question.question):
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Contenido inapropiado")
            response = service.answer_question(question)
            logger.info("Response status: 200")
            return response
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Error en /ask: %s", e)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error interno al responder")

    @app.get("/conversations", summary="Listar historial de conversaciones")
    def list_conversations():
        """
        Devuelve el historial de preguntas y respuestas almacenadas.
        """
        try:
            with storage.db_path.open("rb"):
                import sqlite3
                conn = sqlite3.connect(storage.db_path)
                cursor = conn.execute(
                    "SELECT id, question, answer, context, groundedness, created_at FROM conversations ORDER BY created_at DESC"
                )
                conversations = [
                    {
                        "id": row[0],
                        "question": row[1],
                        "answer": row[2],
                        "context": row[3],
                        "groundedness": row[4],
                        "created_at": row[5],
                    }
                    for row in cursor.fetchall()
                ]
                conn.close()
            return conversations
        except Exception as e:
            logger.error("Error al listar conversaciones: %s", e)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error interno al listar conversaciones"
            )
        
def _contains_sensitive_content(text: str) -> bool:
    sensitive_keywords = ["contraseña", "password", "tarjeta", "credencial", "secreto"]
    return any(keyword in text.lower() for keyword in sensitive_keywords)

def _contains_inappropriate_content(text: str) -> bool:
    inappropriate_keywords = ["odio", "racista", "sexista", "insulto", "ofensivo"]
    return any(keyword in text.lower() for keyword in inappropriate_keywords)