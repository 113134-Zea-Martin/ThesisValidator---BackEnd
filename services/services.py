# ============================================================================
# 1. text_processor.py - Procesamiento y limpieza de texto
# ============================================================================
import re
import logging
from typing import List
from core.config_rag import (
    CHUNK_SIZE, CHUNK_OVERLAP, SMALL_SECTION_THRESHOLD,
    SENTENCE_SPLIT_PATTERN, SECTION_TITLE_PATTERN, 
    SPECIAL_SECTIONS_PATTERN, PAGE_NUMBER_PATTERN, 
    NOISE_LINE_PATTERN, LOG_CHUNK_COUNT
)

logger = logging.getLogger(__name__)

class TextProcessor:
    """Responsable de limpieza y chunking de texto"""
    
    def clean_text(self, text: str) -> str:
        """
        Limpia el texto copiado de PDFs:
        - Elimina números de página sueltos
        - Normaliza espacios y saltos de línea
        - Preserva estructura de secciones
        """
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            
            if not line:
                if cleaned_lines and cleaned_lines[-1] != "":
                    cleaned_lines.append("")
                continue
            
            # Preservar títulos numerados (1. Introducción)
            if re.match(r'^\d+\.(\d+\.)*\s+[A-ZÁÉÍÓÚÑ]', line):
                cleaned_lines.append("\n" + line)
                continue

            # Preservar títulos no numerados (AGRADECIMIENTOS, RESUMEN)
            if re.match(r'^[A-ZÁÉÍÓÚÑ\s]{5,}$', line):
                cleaned_lines.append("\n" + line)
                continue
            
            # Eliminar números de página
            if re.match(PAGE_NUMBER_PATTERN, line):
                continue
            
            # Eliminar líneas de ruido
            if re.match(NOISE_LINE_PATTERN, line):
                continue
            
            cleaned_lines.append(line)
        
        cleaned_text = '\n'.join(cleaned_lines)
        cleaned_text = re.sub(r' +', ' ', cleaned_text)
        cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
        
        return cleaned_text.strip()

    def chunk_text(self, text: str, chunk_size: int = CHUNK_SIZE, 
                   overlap: int = CHUNK_OVERLAP) -> List[str]:
        """
        Chunking mejorado con detección de secciones numeradas y especiales
        """
        section_matches = list(re.finditer(SECTION_TITLE_PATTERN, text))
        special_matches = list(re.finditer(SPECIAL_SECTIONS_PATTERN, text, re.IGNORECASE))
        
        all_matches = section_matches + special_matches
        all_matches.sort(key=lambda m: m.start())
        
        if not all_matches:
            logger.warning("No se detectaron secciones. Usando chunking básico.")
            return self._chunk_basic(text, chunk_size, overlap)
        
        if LOG_CHUNK_COUNT:
            logger.info("✅ Detectadas %d secciones (%d numeradas + %d especiales)", 
                       len(all_matches), len(section_matches), len(special_matches))
        
        chunks = []
        
        for idx, match in enumerate(all_matches):
            section_title = match.group(1).strip()
            section_start = match.end()
            section_end = all_matches[idx + 1].start() if idx + 1 < len(all_matches) else len(text)
            section_content = text[section_start:section_end].strip()
            
            if re.search(SPECIAL_SECTIONS_PATTERN, section_title, re.IGNORECASE):
                logger.info("📌 Sección especial: '%s' (%d chars)", 
                           section_title, len(section_content))
            
            # Secciones pequeñas: mantener completas
            if len(section_content) < chunk_size * SMALL_SECTION_THRESHOLD:
                chunk = f"[Sección: {section_title}]\n\n{section_content}"
                chunks.append(chunk)
            else:
                # Secciones grandes: dividir
                chunks.extend(self._chunk_section(section_title, section_content, chunk_size, overlap))
        
        if LOG_CHUNK_COUNT:
            logger.info("📊 Total chunks generados: %d", len(chunks))
        
        return chunks if chunks else self._chunk_basic(text, chunk_size, overlap)

    def _chunk_section(self, title: str, content: str, chunk_size: int, overlap: int) -> List[str]:
        """Divide una sección grande en múltiples chunks"""
        chunks = []
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        current_chunk = ""
        
        for para in paragraphs:
            if len(current_chunk) + len(para) + 50 > chunk_size:
                if current_chunk:
                    chunks.append(f"[Sección: {title}]\n\n{current_chunk}")
                    overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                    current_chunk = overlap_text + "\n\n" + para
                else:
                    current_chunk = para
            else:
                current_chunk += ("\n\n" if current_chunk else "") + para
        
        if current_chunk:
            chunks.append(f"[Sección: {title}]\n\n{current_chunk}")
        
        return chunks

    def _chunk_basic(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Chunking básico por oraciones"""
        sentences = re.split(SENTENCE_SPLIT_PATTERN, text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return [text] if text.strip() else []
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 <= chunk_size:
                current_chunk += (" " if current_chunk else "") + sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk)
        
        if overlap > 0 and len(chunks) > 1:
            overlapped = [chunks[0]]
            for i in range(1, len(chunks)):
                prev = chunks[i-1]
                curr = chunks[i]
                overlap_text = prev[-overlap:] if len(prev) > overlap else prev
                overlapped.append(overlap_text + " " + curr)
            chunks = overlapped
        
        return chunks


# ============================================================================
# 2. vector_store.py - Gestión de ChromaDB y embeddings
# ============================================================================
import os
import cohere
import chromadb
from pathlib import Path
from typing import List, Dict, Optional
from core.config import config
from core.config_rag import (
    EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE, CHROMA_COLLECTION_NAME,
    CHROMA_DISTANCE_METRIC, MAX_CHUNKS_PER_DOCUMENT, LOG_CHUNK_COUNT
)

logger = logging.getLogger(__name__)

class VectorStore:
    """Responsable de gestionar ChromaDB y generar embeddings"""
    
    def __init__(self, db_path: str = "./data/chroma_db"):
        self.db_path = db_path
        self._chroma_client = None
        self._collection = None
        self._cohere_client = None

    def get_cohere_client(self):
        """Obtiene o crea el cliente de Cohere"""
        if self._cohere_client is None:
            api_key = config.cohere_api_key or os.environ.get("COHERE_API_KEY")
            if not api_key:
                raise EnvironmentError("Cohere API key not configured")
            self._cohere_client = cohere.Client(api_key)
        return self._cohere_client

    def get_collection(self, name: str = CHROMA_COLLECTION_NAME):
        """Obtiene o crea la colección de ChromaDB"""
        if self._chroma_client is None:
            Path(self.db_path).mkdir(parents=True, exist_ok=True)
            self._chroma_client = chromadb.PersistentClient(path=self.db_path)
        
        if self._collection is None:
            self._collection = self._chroma_client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": CHROMA_DISTANCE_METRIC}
            )
        return self._collection

    def reset_collection(self, name: str = CHROMA_COLLECTION_NAME):
        """Elimina la colección actual"""
        try:
            if self._chroma_client is None:
                Path(self.db_path).mkdir(parents=True, exist_ok=True)
                self._chroma_client = chromadb.PersistentClient(path=self.db_path)
            
            self._chroma_client.delete_collection(name=name)
            self._collection = None
            logger.info("✅ Colección '%s' eliminada exitosamente", name)
            return True
        except Exception as e:
            logger.warning("⚠️ Error eliminando colección: %s", e)
            return False

    def list_all_embeddings(self) -> List[Dict]:
        """
        Lista todos los embeddings almacenados en la colección
        """
        try:
            collection = self.get_collection()
            result = collection.get()
            
            embeddings_list = []
            ids = result.get("ids", [])
            metadatas = result.get("metadatas", [])
            
            # Agrupar por documento
            doc_chunks = {}
            for idx, chunk_id in enumerate(ids):
                meta = metadatas[idx] if idx < len(metadatas) else {}
                doc_id = meta.get("source", "unknown")
                
                if doc_id not in doc_chunks:
                    doc_chunks[doc_id] = {
                        "document_id": doc_id,
                        "title": meta.get("title", ""),
                        "chunk_count": 0
                    }
                doc_chunks[doc_id]["chunk_count"] += 1
            
            return list(doc_chunks.values())
            
        except Exception as e:
            logger.error("Error listando embeddings: %s", e)
            return []

    def add_documents(self, document_id: str, chunks: List[str], 
                     title: str = "") -> bool:
        """
        Genera embeddings y almacena chunks en ChromaDB
        """
        if len(chunks) > MAX_CHUNKS_PER_DOCUMENT:
            logger.warning(
                "Documento excede límite de chunks (%d > %d). Truncando.",
                len(chunks), MAX_CHUNKS_PER_DOCUMENT
            )
            chunks = chunks[:MAX_CHUNKS_PER_DOCUMENT]
        
        if LOG_CHUNK_COUNT:
            logger.info("Generados %d chunks para documento %s", len(chunks), document_id)

        collection = self.get_collection()
        ids, documents, metadatas = [], [], []
        
        for idx, chunk in enumerate(chunks):
            chunk_id = f"{document_id}_chunk_{idx}"
            ids.append(chunk_id)
            documents.append(chunk)
            metadatas.append({
                "source": document_id,
                "title": title,
                "chunk_index": idx,
                "chunk_length": len(chunk)
            })

        # Generar embeddings con Cohere si está disponible
        if config.cohere_api_key or os.environ.get("COHERE_API_KEY"):
            co = self.get_cohere_client()
            try:
                for batch_start in range(0, len(documents), EMBEDDING_BATCH_SIZE):
                    batch_end = min(batch_start + EMBEDDING_BATCH_SIZE, len(documents))
                    batch_docs = documents[batch_start:batch_end]
                    batch_ids = ids[batch_start:batch_end]
                    batch_metadatas = metadatas[batch_start:batch_end]
                    
                    logger.info("Procesando lote %d-%d/%d", batch_start, batch_end, len(documents))
                    
                    embeds = co.embed(
                        texts=batch_docs,
                        model=EMBEDDING_MODEL,
                        input_type="search_document",
                        embedding_types=["float"]
                    ).embeddings.float
                    
                    collection.add(
                        ids=batch_ids,
                        documents=batch_docs,
                        metadatas=batch_metadatas,
                        embeddings=embeds,
                    )
                return True
            except Exception:
                logger.exception("Error generando embeddings; usando add sin embeddings")
                collection.add(ids=ids, documents=documents, metadatas=metadatas)
                return False
        else:
            collection.add(ids=ids, documents=documents, metadatas=metadatas)
            return True

    def query(self, query_text: str, n_results: int = 10) -> Dict:
        """
        Busca documentos similares usando embeddings
        """
        co = self.get_cohere_client()
        
        q_embed = co.embed(
            texts=[query_text],
            model=EMBEDDING_MODEL,
            input_type="search_query",
            embedding_types=["float"]
        ).embeddings.float[0]

        collection = self.get_collection()
        
        results = collection.query(
            query_embeddings=[q_embed],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        
        return results


# ============================================================================
# 3. rag_engine.py - Lógica RAG y re-ranking
# ============================================================================
import numpy as np
import cohere
from typing import List, Dict, Optional
from schemas.schemas import AnswerContext
from core.config_rag import (
    CHAT_MODEL, RERANK_MODEL, CHAT_TEMPERATURE, EMBEDDING_MODEL,
    RAG_RETRIEVE_CHUNKS_INITIAL, RAG_RETRIEVE_CHUNKS,
    MAX_CHUNK_DISTANCE, SYSTEM_PROMPT, RAG_PROMPT_TEMPLATE,
    NO_INFO_MESSAGE, LOG_RAG_CONTEXT, LOG_RERANK_SCORES
)

logger = logging.getLogger(__name__)

class RAGEngine:
    """Responsable de la lógica RAG: búsqueda, re-ranking y generación"""
    
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    def should_use_rag(self, question: str) -> bool:
        """
        Usa el LLM para decidir si la pregunta requiere RAG
        """
        co = self.vector_store.get_cohere_client()
        prompt = (
            "Analiza la siguiente pregunta de un usuario. "
            "Si es un saludo, despedida, agradecimiento o pregunta general sobre el sistema, responde SOLO 'NO'. "
            "Si es una pregunta sobre el contenido de los documentos cargados, responde SOLO 'SI'.\n\n"
            f"Pregunta: {question}\nRespuesta:"
        )
        resp = co.chat(
            model=CHAT_MODEL,
            message=prompt,
            temperature=0
        )
        return resp.text.strip().upper().startswith("SI")

    def retrieve_and_rerank(self, question: str, n_final: int = RAG_RETRIEVE_CHUNKS) -> List[Dict]:
        """
        Recupera chunks relevantes y los re-rankea
        """
        # PASO 1: Recuperar candidatos
        results = self.vector_store.query(question, n_results=RAG_RETRIEVE_CHUNKS_INITIAL)
        
        docs = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0] if "distances" in results else None
        ids = results.get("ids", [[]])[0] if "ids" in results else [None] * len(docs)
        
        if not docs:
            return []

        # PASO 2: Re-ranking con Cohere
        co = self.vector_store.get_cohere_client()
        try:
            rerank_response = co.rerank(
                query=question,
                documents=docs,
                model=RERANK_MODEL,
                top_n=n_final
            )
            
            reranked_indices = [result.index for result in rerank_response.results]
            
            if LOG_RERANK_SCORES:
                logger.info("Re-rank: %d chunks -> top %d seleccionados", 
                           len(docs), len(reranked_indices))
                for result in rerank_response.results[:3]:
                    logger.info("  Chunk %d: score %.4f", result.index, result.relevance_score)
            
        except Exception:
            logger.exception("Error en re-ranking; usando orden original")
            reranked_indices = list(range(min(n_final, len(docs))))

        # PASO 3: Construir resultados
        retrieved = []
        seen_chunks = set()
        
        for idx in reranked_indices:
            if idx >= len(docs):
                continue
                
            doc_text = docs[idx]
            raw_id = ids[idx] if idx < len(ids) else None
            
            if raw_id in seen_chunks:
                continue
            seen_chunks.add(raw_id)
            
            meta = metadatas[idx] if idx < len(metadatas) else {}
            dist = distances[idx] if distances and idx < len(distances) else None
            
            # Filtrar por distancia máxima
            if dist is not None and dist > MAX_CHUNK_DISTANCE:
                continue
            
            document_id = None
            if isinstance(meta, dict) and meta.get("source"):
                document_id = str(meta.get("source"))
            elif raw_id and isinstance(raw_id, str) and "_chunk_" in raw_id:
                document_id = raw_id.split("_chunk_")[0]
            
            similarity_score = 0.0
            if dist is not None:
                similarity_score = max(0.0, min(1.0, 1.0 - float(dist)))
            
            retrieved.append({
                "document_id": document_id,
                "chunk_id": raw_id,
                "chunk_index": meta.get("chunk_index") if isinstance(meta, dict) else None,
                "content": doc_text if isinstance(doc_text, str) else "",
                "similarity_score": similarity_score
            })
        
        return retrieved

    def generate_answer(self, question: str, context_items: List[Dict]) -> str:
        """
        Genera respuesta usando el LLM con contexto RAG
        """
        if not context_items:
            return NO_INFO_MESSAGE
        
        context_text = "\n\n".join([
            f"[Fragmento {i+1} de {len(context_items)}]\n{item['content']}" 
            for i, item in enumerate(context_items)
        ])

        prompt = RAG_PROMPT_TEMPLATE.format(
            system_prompt=SYSTEM_PROMPT,
            context=context_text,
            question=question
        )

        co = self.vector_store.get_cohere_client()
        chat_resp = co.chat(
            model=CHAT_MODEL,
            message=prompt,
            temperature=CHAT_TEMPERATURE
        )
        
        return chat_resp.text.strip()

    def calculate_groundedness(self, answer: str, context_chunks: List[str]) -> float:
        """
        Calcula el groundedness (similitud coseno promedio con el contexto)
        """
        if not context_chunks:
            return 0.0
            
        co = self.vector_store.get_cohere_client()
        
        resp_emb = co.embed(
            texts=[answer],
            model=EMBEDDING_MODEL,
            input_type="search_document"
        ).embeddings[0]
        
        ctx_embs = co.embed(
            texts=context_chunks,
            model=EMBEDDING_MODEL,
            input_type="search_document"
        ).embeddings
        
        sims = [
            np.dot(resp_emb, ctx_emb) / (np.linalg.norm(resp_emb) * np.linalg.norm(ctx_emb)) 
            for ctx_emb in ctx_embs
        ]
        
        return float(np.mean(sims)) if sims else 0.0


# ============================================================================
# 4. document_service.py - Servicio principal (REFACTORIZADO)
# ============================================================================
import logging
from typing import List
from core.storage import DocumentStorage
from schemas.schemas import (
    DocumentUpload, DocumentResponse, EmbeddingRequest, EmbeddingResponse,
    SearchQuery, SearchResponse, QuestionRequest, AnswerResponse, AnswerContext
)
from core.config_rag import (
    DEFAULT_SEARCH_RESULTS, CONTENT_SNIPPET_MAX_LENGTH,
    DOCUMENT_SAVED_MESSAGE, EMBEDDINGS_GENERATED_MESSAGE,
    LOG_SEARCH_QUERIES, CHAT_MODEL, SYSTEM_PROMPT, CHAT_TEMPERATURE
)

logger = logging.getLogger(__name__)

class DocumentService:
    """
    Servicio principal que coordina las operaciones con documentos.
    Ahora delega responsabilidades a clases especializadas.
    """
    
    def __init__(self, storage: DocumentStorage):
        self.storage = storage
        self.text_processor = TextProcessor()
        self.vector_store = VectorStore()
        self.rag_engine = RAGEngine(self.vector_store)

    # ========================================================================
    # CRUD de Documentos
    # ========================================================================
    
    def upload_document(self, upload: DocumentUpload) -> DocumentResponse:
        """Guarda un nuevo documento"""
        doc_id = self.storage.save_document(upload.title, upload.content)
        logger.info("Documento %s guardado con %d caracteres", doc_id, len(upload.content))
        return DocumentResponse(message=DOCUMENT_SAVED_MESSAGE, document_id=doc_id)

    def list_documents(self) -> List[DocumentResponse]:
        """Lista todos los documentos"""
        items = self.storage.list_documents()
        return [
            DocumentResponse(message=item["title"], document_id=item["document_id"])
            for item in items
        ]

    # ========================================================================
    # Embeddings
    # ========================================================================
    
    def generate_embedding(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Genera embeddings para un documento"""
        doc = self.storage.get_document(request.document_id)
        if not doc:
            raise ValueError("Documento no encontrado")

        # 1. Limpiar texto
        cleaned_content = self.text_processor.clean_text(doc["content"])
        logger.info("Texto original: %d chars | Texto limpio: %d chars", 
                   len(doc["content"]), len(cleaned_content))

        # 2. Guardar embedding (metadata)
        self.storage.create_embedding(request.document_id, cleaned_content)

        # 3. Chunking
        chunks = self.text_processor.chunk_text(cleaned_content)

        # 4. Almacenar en vector store
        success = self.vector_store.add_documents(
            document_id=request.document_id,
            chunks=chunks,
            title=doc.get("title", "")
        )

        return EmbeddingResponse(message=EMBEDDINGS_GENERATED_MESSAGE)

    def reset_collection(self):
        """Elimina la colección de embeddings"""
        return self.vector_store.reset_collection()

    # ========================================================================
    # Búsqueda
    # ========================================================================
    
    def search_documents(self, query: SearchQuery) -> SearchResponse:
        """Busca documentos similares a la query"""
        k_requested = getattr(query, "k", DEFAULT_SEARCH_RESULTS)
        k_retrieve = min(k_requested * 6, 30)
        
        # Recuperar y re-rankear
        retrieved = self.rag_engine.retrieve_and_rerank(
            query.query, 
            n_final=k_requested
        )
        
        # Construir respuesta
        items = []
        for item in retrieved:
            document_id = item["document_id"]
            
            # Obtener título del documento
            title = ""
            if document_id:
                stored = self.storage.get_document(document_id)
                if stored:
                    title = stored.get("title", "") or ""
            
            content_snippet = item["content"][:CONTENT_SNIPPET_MAX_LENGTH]
            
            items.append({
                "document_id": document_id or item["chunk_id"],
                "title": title,
                "content_snippet": content_snippet,
                "similarity_score": item["similarity_score"],
                "chunk_id": item["chunk_id"],
                "chunk_index": item["chunk_index"]
            })

        if LOG_SEARCH_QUERIES:
            logger.info("Search: '%s' -> %d results (re-ranked)", query.query, len(items))

        return SearchResponse(results=items)

    # ========================================================================
    # Q&A con RAG
    # ========================================================================
    
    def answer_question(self, request: QuestionRequest) -> AnswerResponse:
        """Responde una pregunta usando RAG o directamente del LLM"""
        
        # Decidir si usar RAG
        if not self.rag_engine.should_use_rag(request.question):
            return self._answer_direct(request.question)
        
        # Recuperar contexto relevante
        retrieved = self.rag_engine.retrieve_and_rerank(request.question)
        
        if not retrieved:
            return AnswerResponse(
                question=request.question,
                answer=NO_INFO_MESSAGE,
                context_used=[],
                grounded=False
            )
        
        # Generar respuesta
        answer_text = self.rag_engine.generate_answer(request.question, retrieved)
        
        # Construir contexto usado
        context_used = [
            AnswerContext(
                document_id=item["document_id"],
                chunk_id=item["chunk_id"],
                chunk_index=item["chunk_index"],
                content_snippet=item["content"],
                similarity_score=item["similarity_score"]
            )
            for item in retrieved
        ]
        
        # Calcular groundedness
        context_chunks = [item["content"] for item in retrieved]
        groundedness = self.rag_engine.calculate_groundedness(answer_text, context_chunks)
        
        # Guardar conversación
        self.storage.save_conversation(
            request.question,
            answer_text,
            "\n".join(context_chunks),
            groundedness
        )
        
        return AnswerResponse(
            question=request.question,
            answer=answer_text,
            context_used=context_used,
            grounded=True
        )

    def _answer_direct(self, question: str) -> AnswerResponse:
        """Responde directamente sin RAG (para saludos, etc.)"""
        co = self.vector_store.get_cohere_client()
        prompt = SYSTEM_PROMPT + "\n\n" + question
        
        chat_resp = co.chat(
            model=CHAT_MODEL,
            message=prompt,
            temperature=CHAT_TEMPERATURE
        )
        
        return AnswerResponse(
            question=question,
            answer=chat_resp.text.strip(),
            context_used=[],
            grounded=False
        )