import logging
import os
import re
import cohere
import chromadb
from pathlib import Path
from typing import Dict, List
from core.config import config
from core.storage import DocumentStorage
from schemas.schemas import (
    DocumentUpload, DocumentResponse, EmbeddingRequest, EmbeddingResponse,
    SearchQuery, SearchResponse, QuestionRequest, AnswerResponse, AnswerContext,
)
from core.config_rag import (
    # Modelos
    EMBEDDING_MODEL, CHAT_MODEL, RERANK_MODEL, CHAT_TEMPERATURE,
    # Búsqueda
    DEFAULT_SEARCH_RESULTS, RAG_RETRIEVE_CHUNKS_INITIAL, RAG_RETRIEVE_CHUNKS,
    # Chunking
    CHUNK_SIZE, CHUNK_OVERLAP, CONTENT_SNIPPET_MAX_LENGTH,
    SMALL_SECTION_THRESHOLD, EMBEDDING_BATCH_SIZE,
    # Patrones regex
    SENTENCE_SPLIT_PATTERN, SECTION_TITLE_PATTERN, SPECIAL_SECTIONS_PATTERN,
    PAGE_NUMBER_PATTERN, NOISE_LINE_PATTERN,
    # ChromaDB
    CHROMA_COLLECTION_NAME, CHROMA_DISTANCE_METRIC,
    # Prompts
    SYSTEM_PROMPT, RAG_PROMPT_TEMPLATE, RAG_RERANK_INSTRUCTIONS,
    NO_INFO_MESSAGE, DOCUMENT_SAVED_MESSAGE, EMBEDDINGS_GENERATED_MESSAGE,
    # Logging
    LOG_SEARCH_QUERIES, LOG_CHUNK_COUNT, LOG_RAG_CONTEXT, LOG_RERANK_SCORES,
    # Límites
    MAX_CHUNK_DISTANCE, MAX_CHUNKS_PER_DOCUMENT,
)

logger = logging.getLogger(__name__)

class DocumentService:
    def __init__(self, storage: DocumentStorage):
        self.storage = storage
        self._chroma_client = None
        self._collection = None
        self.chroma_db_path = "./data/chroma_db"

    def upload_document(self, upload: DocumentUpload) -> DocumentResponse:
        doc_id = self.storage.save_document(upload.title, upload.content)
        logger.info("Documento %s guardado con %d caracteres", doc_id, len(upload.content))
        return DocumentResponse(message=DOCUMENT_SAVED_MESSAGE, document_id=doc_id)

    def generate_embedding(self, request: EmbeddingRequest) -> EmbeddingResponse:
        doc = self.storage.get_document(request.document_id)
        if not doc:
            raise ValueError("Documento no encontrado")

        # Limpieza mejorada del texto
        cleaned_content = self._clean_text(doc["content"])
        logger.info("Texto original: %d chars | Texto limpio: %d chars", 
                   len(doc["content"]), len(cleaned_content))

        self.storage.create_embedding(request.document_id, cleaned_content)

        collection = self._get_chroma_collection(name=CHROMA_COLLECTION_NAME)
        
        # Chunking mejorado con más contexto
        chunks = self._chunk_text_improved(
            cleaned_content, 
            chunk_size=CHUNK_SIZE, 
            overlap=CHUNK_OVERLAP
        )
        
        # Validar número máximo de chunks
        if len(chunks) > MAX_CHUNKS_PER_DOCUMENT:
            logger.warning(
                "Documento excede límite de chunks (%d > %d). Truncando.",
                len(chunks), MAX_CHUNKS_PER_DOCUMENT
            )
            chunks = chunks[:MAX_CHUNKS_PER_DOCUMENT]
        
        if LOG_CHUNK_COUNT:
            logger.info("Generados %d chunks para documento %s", len(chunks), request.document_id)
            logger.info("Primer chunk: %s...", chunks[0][:200] if chunks else "VACÍO")
            logger.info("Último chunk: %s...", chunks[-1][:200] if chunks else "VACÍO")

        ids, documents, metadatas = [], [], []
        for idx, chunk in enumerate(chunks):
            cid = f"{request.document_id}_chunk_{idx}"
            ids.append(cid)
            documents.append(chunk)
            metadatas.append({
                "source": request.document_id,
                "title": doc.get("title", ""),
                "chunk_index": idx,
                "chunk_length": len(chunk)
            })

        if config.cohere_api_key or os.environ.get("COHERE_API_KEY"):
            co = self._get_cohere_client()
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
            except Exception:
                logger.exception("Error generando embeddings; usando add sin embeddings")
                collection.add(ids=ids, documents=documents, metadatas=metadatas)
        else:
            collection.add(ids=ids, documents=documents, metadatas=metadatas)

        return EmbeddingResponse(message=EMBEDDINGS_GENERATED_MESSAGE)

    def _clean_text(self, text: str) -> str:
        """
        Limpia el texto copiado de PDFs usando patrones de config_rag:
        - Elimina números de página sueltos
        - Normaliza espacios y saltos de línea
        - Preserva estructura de secciones
        """
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            
            # Saltar líneas vacías duplicadas
            if not line:
                if cleaned_lines and cleaned_lines[-1] != "":
                    cleaned_lines.append("")
                continue
            
            # Detectar y preservar títulos de secciones
            # Detectar títulos numerados (1. Introducción)
            if re.match(r'^\d+\.(\d+\.)*\s+[A-ZÁÉÍÓÚÑ]', line):
                cleaned_lines.append("\n" + line)
                continue

            # Detectar títulos no numerados típicos de tesis (AGRADECIMIENTOS, RESUMEN, etc.)
            if re.match(r'^[A-ZÁÉÍÓÚÑ\s]{5,}$', line):
                cleaned_lines.append("\n" + line)
                continue

            
            # Eliminar números de página usando patrón del config
            if re.match(PAGE_NUMBER_PATTERN, line):
                continue
            
            # Eliminar líneas de ruido usando patrón del config
            if re.match(NOISE_LINE_PATTERN, line):
                continue
            
            cleaned_lines.append(line)
        
        # Unir líneas
        cleaned_text = '\n'.join(cleaned_lines)
        
        # Normalizar espacios múltiples
        cleaned_text = re.sub(r' +', ' ', cleaned_text)
        
        # Normalizar saltos de línea múltiples (máximo 2)
        cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
        
        return cleaned_text.strip()

    def _chunk_text_improved(self, text: str, chunk_size: int = CHUNK_SIZE, 
                           overlap: int = CHUNK_OVERLAP) -> List[str]:
        """
        Chunking mejorado con detección de secciones numeradas Y especiales
        """
        # Detectar secciones numeradas (ej: "10. Conclusiones")
        section_matches = list(re.finditer(SECTION_TITLE_PATTERN, text))
        
        # Detectar secciones especiales sin número (ej: "Agradecimientos")
        special_matches = list(re.finditer(SPECIAL_SECTIONS_PATTERN, text, re.IGNORECASE))
        
        # Combinar y ordenar por posición en el texto
        all_matches = section_matches + special_matches
        all_matches.sort(key=lambda m: m.start())
        
        if not all_matches:
            logger.warning("No se detectaron secciones. Usando chunking básico.")
            return self._chunk_text_basic(text, chunk_size, overlap)
        
        if LOG_CHUNK_COUNT:
            logger.info("✅ Detectadas %d secciones (%d numeradas + %d especiales)", 
                       len(all_matches), len(section_matches), len(special_matches))
        
        chunks = []
        
        for idx, match in enumerate(all_matches):
            section_title = match.group(1).strip()
            section_start = match.end()
            
            # Determinar dónde termina esta sección
            if idx + 1 < len(all_matches):
                section_end = all_matches[idx + 1].start()
            else:
                section_end = len(text)
            
            section_content = text[section_start:section_end].strip()
            
            # Log especial para secciones sin número
            if re.search(SPECIAL_SECTIONS_PATTERN, section_title, re.IGNORECASE):
                logger.info("📌 Sección especial: '%s' (%d chars)", 
                           section_title, len(section_content))
            
            # Secciones pequeñas: mantener completas
            if len(section_content) < chunk_size * SMALL_SECTION_THRESHOLD:
                chunk = f"[Sección: {section_title}]\n\n{section_content}"
                chunks.append(chunk)
            else:
                # Secciones grandes: dividir en múltiples chunks
                paragraphs = [p.strip() for p in section_content.split('\n\n') if p.strip()]
                
                current_chunk = ""
                for para in paragraphs:
                    if len(current_chunk) + len(para) + 50 > chunk_size:
                        if current_chunk:
                            chunk = f"[Sección: {section_title}]\n\n{current_chunk}"
                            chunks.append(chunk)
                            
                            # Overlap inteligente
                            overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                            current_chunk = overlap_text + "\n\n" + para
                        else:
                            current_chunk = para
                    else:
                        current_chunk += ("\n\n" if current_chunk else "") + para
                
                # Agregar último chunk de la sección
                if current_chunk:
                    chunk = f"[Sección: {section_title}]\n\n{current_chunk}"
                    chunks.append(chunk)
        
        if LOG_CHUNK_COUNT:
            logger.info("📊 Total chunks generados: %d", len(chunks))
        
        return chunks if chunks else self._chunk_text_basic(text, chunk_size, overlap)

    def _chunk_text_basic(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Chunking básico por oraciones usando SENTENCE_SPLIT_PATTERN"""
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
        
        # Agregar overlap
        if overlap > 0 and len(chunks) > 1:
            overlapped = [chunks[0]]
            for i in range(1, len(chunks)):
                prev = chunks[i-1]
                curr = chunks[i]
                overlap_text = prev[-overlap:] if len(prev) > overlap else prev
                overlapped.append(overlap_text + " " + curr)
            chunks = overlapped
        
        return chunks

    def _get_cohere_client(self):
        api_key = config.cohere_api_key or os.environ.get("COHERE_API_KEY")
        if not api_key:
            raise EnvironmentError("Cohere API key not configured")
        return cohere.Client(api_key)

    def _get_chroma_collection(self, name: str = None):
        if name is None:
            name = CHROMA_COLLECTION_NAME
        if self._chroma_client is None:
            Path(self.chroma_db_path).mkdir(parents=True, exist_ok=True)
            self._chroma_client = chromadb.PersistentClient(path=self.chroma_db_path)
        if self._collection is None:
            self._collection = self._chroma_client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": CHROMA_DISTANCE_METRIC}
            )
        return self._collection

    def reset_collection(self):
        """Elimina la colección actual para regenerar embeddings desde cero"""
        try:
            if self._chroma_client is None:
                Path(self.chroma_db_path).mkdir(parents=True, exist_ok=True)
                self._chroma_client = chromadb.PersistentClient(path=self.chroma_db_path)
            
            self._chroma_client.delete_collection(name=CHROMA_COLLECTION_NAME)
            self._collection = None
            logger.info("✅ Colección '%s' eliminada exitosamente", CHROMA_COLLECTION_NAME)
            return True
        except Exception as e:
            logger.warning("⚠️ Error eliminando colección: %s", e)
            return False

    def search_documents(self, query: SearchQuery) -> SearchResponse:
        co = self._get_cohere_client()
        
        try:
            q_embed = co.embed(
                texts=[query.query],
                model=EMBEDDING_MODEL,
                input_type="search_query",
                embedding_types=["float"]
            ).embeddings.float[0]
        except Exception:
            logger.exception("Error al generar embedding de la query")
            raise

        collection = self._get_chroma_collection()
        
        # PASO 1: Recuperar más candidatos (3x lo solicitado para re-rankear)
        k_requested = getattr(query, "k", DEFAULT_SEARCH_RESULTS)
        k_retrieve = min(k_requested * 6, 30)  # Recuperar 6x más, máximo 30
        
        try:
            results = collection.query(
                query_embeddings=[q_embed],
                n_results=k_retrieve,
                include=["documents", "metadatas", "distances"]
            )
            
            docs = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0] if "distances" in results else None
            ids = results.get("ids", [[]])[0] if "ids" in results else [None] * len(docs)
            
        except Exception:
            logger.exception("Error consultando Chroma")
            raise

        if not docs:
            return SearchResponse(results=[])

        # PASO 2: Re-ranking con Cohere (igual que en answer_question)
        try:
            rerank_response = co.rerank(
                query=query.query,
                documents=docs,
                model=RERANK_MODEL,
                top_n=k_requested  # Solo devolver los K solicitados
            )
            
            # Reordenar según el re-ranking
            reranked_indices = [result.index for result in rerank_response.results]
            
            if LOG_RERANK_SCORES:
                logger.info("Search re-rank: %d chunks -> top %d seleccionados", 
                           len(docs), len(reranked_indices))
            
        except Exception:
            logger.exception("Error en re-ranking de search; usando orden original")
            reranked_indices = list(range(min(k_requested, len(docs))))

        # PASO 3: Construir resultados con chunks re-ranqueados
        items = []
        for idx in reranked_indices:
            if idx >= len(docs):
                continue
                
            doc_text = docs[idx]
            raw_id = ids[idx] if idx < len(ids) else None
            meta = metadatas[idx] if idx < len(metadatas) else {}
            
            document_id = None
            if isinstance(meta, dict) and meta.get("source"):
                document_id = str(meta.get("source"))
            elif raw_id and isinstance(raw_id, str) and "_chunk_" in raw_id:
                document_id = raw_id.split("_chunk_")[0]
            
            content_snippet = (doc_text[:CONTENT_SNIPPET_MAX_LENGTH]) if isinstance(doc_text, str) else ""
            
            title = ""
            if document_id:
                stored = self.storage.get_document(document_id)
                if stored:
                    title = stored.get("title", "") or ""
            
            # Usar distancia de Chroma para el score (aunque re-ranking ordenó)
            similarity_score = 0.0
            dist = distances[idx] if distances and idx < len(distances) else None
            if dist is not None:
                try:
                    similarity_score = max(0.0, min(1.0, 1.0 - float(dist)))
                except Exception:
                    pass
            
            items.append({
                "document_id": document_id or (raw_id or f"chunk_{i}"),
                "title": title,
                "content_snippet": content_snippet,
                "similarity_score": similarity_score,
                "chunk_id": raw_id,
                "chunk_index": meta.get("chunk_index") if isinstance(meta, dict) else None
            })

        if LOG_SEARCH_QUERIES:
            logger.info("Search: '%s' -> %d results (re-ranked)", query.query, len(items))

        return SearchResponse(results=items)

    def answer_question(self, request: QuestionRequest) -> AnswerResponse:
        co = self._get_cohere_client()

        # PASO 1: Generar embedding de la pregunta
        try:
            q_embed = co.embed(
                texts=[request.question],
                model=EMBEDDING_MODEL,
                input_type="search_query",
                embedding_types=["float"]
            ).embeddings.float[0]
        except Exception:
            logger.exception("Error al generar embedding de la pregunta")
            raise

        collection = self._get_chroma_collection()
        
        # PASO 2: Recuperar chunks candidatos (más de los que necesitamos finalmente)
        try:
            results = collection.query(
                query_embeddings=[q_embed],
                n_results=RAG_RETRIEVE_CHUNKS_INITIAL,  # Recuperar 30 candidatos
                include=["documents", "metadatas", "distances"]
            )
            
            docs = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0] if "distances" in results else None
            ids = results.get("ids", [[]])[0] if "ids" in results else [None] * len(docs)
            
        except Exception:
            logger.exception("Error consultando Chroma para RAG")
            raise

        if not docs:
            return AnswerResponse(
                question=request.question,
                answer=NO_INFO_MESSAGE,
                context_used=[],
                grounded=False
            )

        # PASO 3: Re-ranking con Cohere (seleccionar los mejores 8)
        try:
            rerank_response = co.rerank(
                query=request.question,
                documents=docs,
                model=RERANK_MODEL,
                top_n=RAG_RETRIEVE_CHUNKS  # Solo los mejores 8
            )
            
            # Reordenar según el re-ranking
            reranked_indices = [result.index for result in rerank_response.results]
            
            if LOG_RERANK_SCORES:
                logger.info("Re-rank: %d chunks -> top %d seleccionados", 
                           len(docs), len(reranked_indices))
                for result in rerank_response.results[:3]:  # Log top 3
                    logger.info("  Chunk %d: score %.4f", result.index, result.relevance_score)
            
        except Exception:
            logger.exception("Error en re-ranking; usando orden original")
            reranked_indices = list(range(min(RAG_RETRIEVE_CHUNKS, len(docs))))

        # PASO 4: Construir contexto con chunks re-ranqueados
        context_used = []
        seen_chunks = set()
        
        for idx in reranked_indices:
            if idx >= len(docs):
                continue
                
            doc_text = docs[idx]
            raw_id = ids[idx] if idx < len(ids) else None
            
            # Evitar duplicados
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
            
            context_item = AnswerContext(
                document_id=document_id,
                chunk_id=raw_id,
                chunk_index=meta.get("chunk_index") if isinstance(meta, dict) else None,
                content_snippet=doc_text if isinstance(doc_text, str) else "",
                similarity_score=similarity_score
            )
            context_used.append(context_item)

        # PASO 5: Construir prompt con instrucciones de re-ranking
        context_text = "\n\n".join([
            f"[Fragmento {i+1} de {len(context_used)}]\n{c.content_snippet}" 
            for i, c in enumerate(context_used)
        ])
        
        prompt = RAG_PROMPT_TEMPLATE.format(
            system_prompt=SYSTEM_PROMPT,
            context=context_text,
            question=request.question
        )

        # PASO 6: Generar respuesta
        try:
            chat_resp = co.chat(
                model=CHAT_MODEL,
                message=prompt,
                temperature=CHAT_TEMPERATURE
            )
            answer_text = chat_resp.text.strip()
        except Exception:
            logger.exception("Error llamando a la API de chat")
            raise

        grounded = len(context_used) > 0

        if LOG_RAG_CONTEXT:
            logger.info("Question: '%s' | Grounded: %s | Context: %d chunks", 
                       request.question, grounded, len(context_used))

        return AnswerResponse(
            question=request.question,
            answer=answer_text,
            context_used=context_used,
            grounded=grounded
        )

    def list_documents(self) -> List[DocumentResponse]:
        items = self.storage.list_documents()
        return [
            DocumentResponse(message=item["title"], document_id=item["document_id"])
            for item in items
        ]