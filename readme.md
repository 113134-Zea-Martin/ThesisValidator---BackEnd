# Get Talent RAG API

Sistema RAG (Retrieval-Augmented Generation) para búsqueda semántica y generación de respuestas fundamentadas sobre documentos académicos, especialmente tesis doctorales en Astronomía.

---

## Tabla de contenidos

- [Características](#características)
- [Arquitectura](#arquitectura)
- [Decisiones técnicas](#decisiones-técnicas)
- [Trade-offs](#trade-offs)
- [Limitaciones](#limitaciones)
- [Despliegue](#despliegue)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Instalación](#instalación)
- [Uso](#uso)
- [Ejemplo de flujo](#ejemplo-de-flujo)
- [Tecnologías utilizadas](#tecnologías-utilizadas)
- [Notas](#notas)

---

## Características

- **Carga de documentos**: Sube documentos académicos en texto plano.
- **Generación de embeddings**: Indexa los documentos usando embeddings semánticos (Cohere + ChromaDB).
- **Búsqueda semántica**: Encuentra fragmentos relevantes usando búsqueda vectorial y re-ranking.
- **Preguntas y respuestas (RAG)**: Realiza preguntas en lenguaje natural y recibe respuestas fundamentadas en el contenido cargado.
- **Historial de conversaciones**: Guarda y consulta el historial de preguntas y respuestas.
- **API RESTful**: Implementada con FastAPI.

---

## Arquitectura

El sistema sigue una arquitectura modular basada en servicios:

- **API REST**: FastAPI expone endpoints para carga, búsqueda, embeddings y preguntas.
- **Procesamiento de texto**: Limpieza y chunking de documentos para mejorar la indexación.
- **Vector Store**: ChromaDB almacena los embeddings y permite búsquedas vectoriales eficientes.
- **Embeddings y LLM**: Cohere se utiliza tanto para generar embeddings como para la generación de respuestas y re-ranking.
- **RAG Engine**: Orquesta la recuperación de contexto relevante y la generación de respuestas fundamentadas.
- **Persistencia**: SQLite almacena documentos y el historial de conversaciones.

---

## Decisiones técnicas

### ¿Por qué Cohere?

- **Calidad multilingüe**: Los modelos de Cohere ofrecen embeddings y generación de texto de alta calidad en español y otros idiomas.
- **Facilidad de integración**: API simple y bien documentada.
- **Modelos especializados**: Permite usar modelos distintos para embeddings, chat y re-ranking.

### ¿Por qué ChromaDB?

- **Vector store eficiente**: ChromaDB es rápido, escalable y fácil de usar para almacenar y consultar embeddings.
- **Persistencia local**: Permite almacenamiento en disco, ideal para prototipos y despliegues sencillos.
- **Integración directa**: Compatible con los formatos de embeddings generados por Cohere.

### ¿Por qué re-ranking?

- **Precisión**: El re-ranking semántico (Cohere Rerank) permite seleccionar los fragmentos más relevantes para la pregunta, mejorando la calidad de las respuestas.
- **Reducción de contexto**: Permite limitar el contexto enviado al LLM, optimizando costos y tiempos de respuesta.

### ¿Por qué decisión dinámica de RAG?

- **Eficiencia**: No todas las preguntas requieren buscar en los documentos. El sistema decide si usar RAG o responder directamente, ahorrando recursos.
- **Mejor experiencia de usuario**: Saludos, agradecimientos o preguntas generales reciben respuestas rápidas y adecuadas.

---

## Trade-offs

- **Dependencia de servicios externos**: El sistema depende de Cohere para embeddings y generación, lo que implica costos y posibles límites de uso.
- **Persistencia local**: ChromaDB y SQLite funcionan localmente, lo que limita la escalabilidad en producción.
- **Chunking heurístico**: El chunking por secciones y oraciones puede no ser óptimo para todos los documentos, pero es simple y robusto.

---

## Limitaciones

- **Escalabilidad**: El almacenamiento local limita el manejo de grandes volúmenes de documentos o usuarios concurrentes.
- **Calidad dependiente del input**: Documentos mal estructurados o con mucho ruido pueden afectar la calidad de los embeddings y las respuestas.
- **Cobertura de modelos**: Cohere puede no cubrir todos los matices del lenguaje académico o técnico en español.
- **No hay control de acceso**: Actualmente, la API no implementa autenticación ni autorización.

---

## Despliegue

El sistema puede ejecutarse localmente mediante FastAPI y Uvicorn.
La arquitectura es compatible con despliegues en plataformas como Render o Railway.


---

## Estructura del proyecto

```
.
├── main.py
├── requirements.txt
├── readme.md
├── api/
│   └── endpoints.py
├── core/
│   ├── config.py
│   ├── config_rag.py
│   ├── storage.py
│   └── .env
├── data/
│   └── chroma_db/
├── schemas/
│   └── schemas.py
├── services/
│   └── services.py
```

---

## Instalación

1. **Clona el repositorio**  
   ```sh
   git clone <repo-url>
   cd ThesisValidator---BackEnd
   ```

2. **Crea un entorno virtual**  
   ```sh
   python -m venv .venv
   source .venv/bin/activate  # En Windows: .venv\Scripts\activate
   ```

3. **Instala las dependencias**  
   ```sh
   pip install -r requirements.txt
   ```

4. **Configura las variables de entorno**  
   - Edita `core/.env` y coloca tu API Key de Cohere:
     ```
     COHERE_API_KEY=tu_api_key
     ```

---

## Uso

1. **Inicia el servidor**
   ```sh
   uvicorn main:app --reload
   ```

2. **Accede a la documentación interactiva**
   - Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
   - Redoc: [http://localhost:8000/redoc](http://localhost:8000/redoc)

3. **Endpoints principales**
   - `POST /upload`: Sube un documento académico.
   - `POST /generate-embeddings`: Genera embeddings para un documento.
   - `POST /search`: Busca fragmentos relevantes.
   - `POST /ask`: Realiza una pregunta sobre los documentos cargados.
   - `GET /documents`: Lista los documentos cargados.
   - `GET /embeddings`: Lista los embeddings generados.
   - `GET /conversations`: Historial de preguntas y respuestas.
   - `GET /health`: Chequeo de salud del servicio.

---

## Ejemplo de flujo

1. **Subir documento**
   ```json
   POST /upload
   {
     "title": "Tesis sobre Supernovas",
     "content": "Texto completo de la tesis..."
   }
   ```

2. **Generar embeddings**
   ```json
   POST /generate-embeddings
   {
     "document_id": "1"
   }
   ```

3. **Buscar**
   ```json
   POST /search
   {
     "query": "¿Qué es una supernova?"
   }
   ```

4. **Preguntar**
   ```json
   POST /ask
   {
     "question": "¿Cuáles son los tipos de supernovas?"
   }
   ```

---

## Tecnologías utilizadas

- [FastAPI](https://fastapi.tiangolo.com/)
- [Cohere](https://cohere.com/) (embeddings y generación)
- [ChromaDB](https://www.trychroma.com/) (vector store)
- [SQLite](https://www.sqlite.org/) (almacenamiento persistente)
- [Pydantic](https://docs.pydantic.dev/) (validación de datos)

---

## Notas

- El sistema está optimizado para documentos académicos en español, pero soporta textos multilingües.
- Las respuestas siempre se generan en español, con tono académico y sin inventar información.
- El almacenamiento de documentos y conversaciones es local (SQLite y archivos en `data/`).

---

Desarrollado para Get Talent - Tesis Validator RAG API.