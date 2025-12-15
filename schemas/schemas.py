from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

# ========== REQUEST ==========

class DocumentUpload(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    content: str = Field(..., min_length=1)

class EmbeddingRequest(BaseModel):
    document_id: str = Field(..., min_length=1)

class SearchQuery(BaseModel):
    query: str = Field(..., min_length=1)

class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1)

# ========== RESPONSE ==========
class DocumentResponse(BaseModel):
    message: str
    document_id: str

class EmbeddingResponse(BaseModel):
    message: str

class SearchResultItem(BaseModel):
    document_id: str
    title: str
    content_snippet: str
    similarity_score: float = Field(..., ge=0.0, le=1.0)

class SearchResponse(BaseModel):
    results: List[SearchResultItem] = []


# ...existing code...

class AnswerContext(BaseModel):
    """Contexto individual usado para generar una respuesta"""
    document_id: Optional[str] = None
    chunk_id: Optional[str] = None
    chunk_index: Optional[int] = None
    content_snippet: str
    similarity_score: float

    class Config:
        json_schema_extra = {
            "example": {
                "document_id": "doc_123",
                "chunk_id": "doc_123_chunk_5",
                "chunk_index": 5,
                "content_snippet": "Texto del fragmento...",
                "similarity_score": 0.89
            }
        }


class AnswerResponse(BaseModel):
    """Respuesta a una pregunta con contexto utilizado"""
    question: str
    answer: str
    context_used: Optional[List[AnswerContext]] = None  # Cambiar de None a lista vacía por defecto
    grounded: bool

    class Config:
        json_schema_extra = {
            "example": {
                "question": "¿Qué es una supernova?",
                "answer": "Una supernova es...",
                "context_used": [
                    {
                        "document_id": "doc_1",
                        "chunk_id": "doc_1_chunk_0",
                        "chunk_index": 0,
                        "content_snippet": "Las supernovas son...",
                        "similarity_score": 0.95
                    }
                ],
                "grounded": True
            }
        }

# ...existing code...

class ErrorResponse(BaseModel):
    error: str
    timestamp: datetime = datetime.utcnow()