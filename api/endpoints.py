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
    """
    Registra todos los endpoints de la API para la gestión de documentos, embeddings, búsqueda y preguntas.
    """
    storage = DocumentStorage()
    service = DocumentService(storage)

    @app.post(
        "/upload",
        response_model=DocumentResponse,
        status_code=status.HTTP_201_CREATED,
        summary="Cargar documento",
        description="Carga un nuevo documento al sistema. El documento será almacenado y estará disponible para procesamiento y búsqueda."
    )
    def upload_document(document: DocumentUpload):
        """
        Sube un documento al sistema.

        - **title**: Título del documento.
        - **content**: Contenido completo del documento (texto plano).
        """
        try:
            return service.upload_document(document)
        except Exception as e:
            logger.error("Error en /upload: %s", e)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error interno al cargar el documento")
        
    @app.get(
        "/documents",
        response_model=List[DocumentResponse],
        summary="Listar documentos",
        description="Devuelve una lista de todos los documentos cargados en el sistema, incluyendo su identificador y título."
    )
    def list_documents():
        """
        Lista todos los documentos almacenados.

        Devuelve una lista de objetos con el ID y el título de cada documento.
        """
        try:
            documents = service.list_documents()
            return documents
        except Exception as e:
            logger.error("Error en /documents: %s", e)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error interno al listar documentos")
        
    @app.post(
        "/generate-embeddings",
        response_model=EmbeddingResponse,
        summary="Generar embeddings",
        description="Genera los embeddings para un documento previamente cargado. Esto permite que el documento sea indexado y utilizado en búsquedas semánticas y respuestas a preguntas."
    )
    def generate_embeddings(request: EmbeddingRequest):
        """
        Genera embeddings para un documento.

        - **document_id**: ID del documento para el que se generarán los embeddings.
        """
        try:
            return service.generate_embedding(request)
        except ValueError:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Documento no encontrado")
        except Exception as e:
            logger.error("Error en /generate-embeddings: %s", e)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error interno al generar embeddings")

    @app.get(
        "/embeddings",
        summary="Listar embeddings generados en ChromaDB",
        description="Devuelve información sobre los embeddings generados y almacenados en la base vectorial (ChromaDB). Muestra cuántos chunks tiene cada documento indexado."
    )
    def _list_embeddings():
        """
        Lista los embeddings generados en la base vectorial.

        Devuelve una lista con el ID del documento, título y cantidad de chunks indexados para cada documento.
        """
        try:
            return service.vector_store.list_all_embeddings()
        except Exception as e:
            logger.error("Error al listar embeddings: %s", e)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                detail="Error interno al listar embeddings"
            )

    @app.post(
        "/search",
        response_model=SearchResponse,
        summary="Buscar documentos",
        description="Realiza una búsqueda semántica sobre los documentos indexados utilizando embeddings. Devuelve los fragmentos más relevantes para la consulta."
    )
    def search_documents(query: SearchQuery):
        """
        Busca fragmentos relevantes en los documentos indexados.

        - **query**: Texto de la consulta de búsqueda.
        - **k** (opcional): Número de resultados a devolver (por defecto, valor configurado en el sistema).

        Devuelve una lista de fragmentos relevantes, con información sobre el documento, el fragmento y la puntuación de similitud.
        """
        try:
            if _contains_sensitive_content(query.query):
                logger.warning("Consulta sensible detectada")
                return SearchResponse(results=[])
            return service.search_documents(query)
        except Exception as e:
            logger.error("Error en /search: %s", e)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error interno en la búsqueda")

    @app.post(
        "/ask",
        response_model=AnswerResponse,
        summary="Hacer pregunta",
        description="Permite realizar una pregunta sobre el contenido de los documentos cargados. El sistema utiliza RAG (Retrieval-Augmented Generation) para buscar información relevante y generar una respuesta fundamentada."
    )
    def ask_question(question: QuestionRequest):
        """
        Realiza una pregunta sobre los documentos cargados.

        - **question**: Pregunta en lenguaje natural.

        El sistema buscará fragmentos relevantes y generará una respuesta fundamentada en el contenido de los documentos.
        """
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

    @app.get(
        "/conversations",
        summary="Listar historial de conversaciones",
        description="Devuelve el historial de preguntas y respuestas realizadas al sistema, incluyendo el contexto utilizado y el groundedness de cada respuesta."
    )
    def list_conversations():
        """
        Lista el historial de conversaciones (preguntas y respuestas).

        Devuelve una lista de preguntas realizadas, respuestas generadas, contexto utilizado y groundedness para cada interacción.
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
    """
    Detecta si el texto contiene palabras sensibles (por ejemplo, contraseñas o datos privados).
    """
    sensitive_keywords = ["contraseña", "password", "tarjeta", "credencial", "secreto"]
    return any(keyword in text.lower() for keyword in sensitive_keywords)

def _contains_inappropriate_content(text: str) -> bool:
    """
    Detecta si el texto contiene palabras inapropiadas (odio, insultos, etc.).
    """
    inappropriate_keywords = ["odio", "racista", "sexista", "insulto", "ofensivo"]
    return any(keyword in text.lower() for keyword in inappropriate_keywords)