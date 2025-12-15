# config_rag.py
"""
Configuración centralizada para el sistema RAG académico.
Optimizado para análisis y consulta de tesis doctorales en Astronomía.
"""

# ==================== CONFIGURACIÓN DE MODELOS ====================

# Modelo de embeddings de Cohere (multilingüe, apto para textos académicos)
EMBEDDING_MODEL = "embed-multilingual-v3.0"

# Modelo de chat/generación de Cohere
CHAT_MODEL = "command-a-03-2025"

# Modelo de re-ranking de Cohere para RAG
RERANK_MODEL = "rerank-v3.5"

# Temperatura baja para respuestas deterministas
CHAT_TEMPERATURE = 0.0

# ==================== CONFIGURACIÓN DE BÚSQUEDA ====================

# Número de resultados por defecto en búsquedas
DEFAULT_SEARCH_RESULTS = 5

# Tamaño de cada chunk en caracteres
CHUNK_SIZE = 1500

# Superposición entre chunks en caracteres
CHUNK_OVERLAP = 300

# Número de chunks a recuperar ANTES de re-ranking (candidatos amplios)
RAG_RETRIEVE_CHUNKS_INITIAL = 30

# Número de chunks finales después de re-ranking
RAG_RETRIEVE_CHUNKS = 8

# Longitud máxima del snippet de contenido (caracteres)
CONTENT_SNIPPET_MAX_LENGTH = 3000

# Umbral máximo de distancia para considerar un chunk relevante
MAX_CHUNK_DISTANCE = 1.0

# Porcentaje del chunk_size para considerar una sección "pequeña" (mantener completa)
SMALL_SECTION_THRESHOLD = 0.8

# ==================== CONFIGURACIÓN DE CHUNKING ====================

# Patrón regex para dividir texto en oraciones
SENTENCE_SPLIT_PATTERN = r'(?<=[\.\!\?\;\:])\s+'

# Patrón regex para detectar títulos de secciones
# Ej: "10. Conclusiones", "8.3. Observaciones"
# SECTION_TITLE_PATTERN = r'\n(\d+\.\s+[A-ZÁÉÍÓÚÑ][^\n]{5,80})\n'
SECTION_TITLE_PATTERN = (
    r'(?:^|\n)'
    r'('
    r'\d+\.(?:\d+\.)*\s+[^\n]+'      # 1. Introducción / 9.4 Resultados
    r'\n(\d+\.\s+[A-ZÁÉÍÓÚÑ][^\n]{5,80})\n'
    r'|AGRADECIMIENTOS'
    r'|Agradecimientos'
    r'|RESUMEN'
    r'|Resumen'
    r'|ABSTRACT'
    r'|CONCLUSIONES'
    r'|Conclusiones'
    r')'
)

SPECIAL_SECTIONS_PATTERN = r'\n((?:Agradecimientos?|Dedicatoria|Resumen|Abstract|Prólogo|Prefacio|Introducción\s+General)[^\n]*)\n'


# Patrón regex para detectar números de página sueltos
PAGE_NUMBER_PATTERN = r'^\d{1,3}$'

# Patrón regex para líneas de solo puntos/guiones
# NOISE_LINE_PATTERN = r'^[\.\-\s]+$'
NOISE_LINE_PATTERN = (
    r'^(\.{3,}|_{3,}|-{3,}|\s*)$'
)


# ==================== CONFIGURACIÓN DE CHROMA ====================

# Nombre de la colección en ChromaDB
CHROMA_COLLECTION_NAME = "academic_documents"

# Métrica de distancia para ChromaDB
CHROMA_DISTANCE_METRIC = "cosine"

# Tamaño de lote para embeddings (límite API de Cohere)
EMBEDDING_BATCH_SIZE = 90

# ==================== PROMPTS Y MENSAJES ====================

# Prompt del sistema para el asistente RAG académico
SYSTEM_PROMPT = (
    "[SYSTEM PROMPT – RESTRICCIÓN PERMANENTE]\n"
    "Eres un asistente especializado en el análisis y explicación de documentos académicos "
    "y científicos, con foco en tesis doctorales en Astronomía.\n"
    "Tu objetivo es brindar respuestas precisas, claras y verificables, basadas exclusivamente "
    "en la información provista en el contexto.\n"
    "No inventes datos, no hagas suposiciones y no completes información faltante.\n"
    "Si la respuesta no se encuentra en el contexto, debes indicarlo explícitamente.\n"
    "Responde siempre en español, sin usar emojis, con un tono académico y explicativo.\n"
    "La misma pregunta debe generar siempre la misma respuesta cuando el contexto sea el mismo."
    "Si la pregunta es un saludo, despido o no está relacionada con el contexto, responde cortésmente que "
    "estás aquí para ayudar con consultas sobre la tesis doctoral proporcionada.\n"
)

# Instrucciones adicionales para el prompt con re-ranking
RAG_RERANK_INSTRUCTIONS = (
    "IMPORTANTE: Los fragmentos fueron seleccionados específicamente para tu pregunta mediante re-ranking semántico.\n"
    "Si encuentras información relevante distribuida en varios fragmentos, sintetízala coherentemente.\n"
)

# Plantilla para construir el prompt de RAG
RAG_PROMPT_TEMPLATE = """{system_prompt}

[INSTRUCCIONES DE ROL]
Actúa como un asistente académico que ayuda a comprender una tesis doctoral en Astronomía.
Explica conceptos técnicos de forma clara y neutral, manteniendo precisión científica.
No agregues información externa ni interpretaciones personales.

[REGLAS DE SEGURIDAD]
No generes información sensible, privada o especulativa.
No inventes resultados, conclusiones ni afirmaciones no presentes en el texto.
Si la información solicitada no está en el contexto, indícalo claramente.

[REGLAS DE GROUNDING – RAG]
Usa exclusivamente la información contenida dentro del bloque <CONTEXT>.
No mezcles conocimiento previo del modelo con el contenido del documento.
Si el contexto no responde la consulta, responde exactamente:
"El contexto no provee esa información."

[CONTEXT]
{context}
[/CONTEXT]

Pregunta del usuario:
"{question}"

"""

# Mensaje cuando no hay información disponible
NO_INFO_MESSAGE = "El contexto no provee esa información."

# Mensaje de éxito al guardar documento
DOCUMENT_SAVED_MESSAGE = "Documento académico guardado correctamente."

# Mensaje de éxito al generar embeddings
EMBEDDINGS_GENERATED_MESSAGE = "Embeddings y fragmentos indexados correctamente."

# ==================== CONFIGURACIÓN DE LOGGING ====================

# Nivel de detalle en los logs
LOG_SEARCH_QUERIES = True
LOG_CHUNK_COUNT = True
LOG_RAG_CONTEXT = True
LOG_RERANK_SCORES = True  # Nuevo: log de scores de re-ranking

# ==================== LÍMITES Y VALIDACIONES ====================

# Longitud mínima de una query de búsqueda
MIN_QUERY_LENGTH = 1

# Longitud máxima de una query de búsqueda
MAX_QUERY_LENGTH = 500

# Longitud mínima del contenido de un documento
MIN_DOCUMENT_CONTENT_LENGTH = 100

# Número máximo de chunks por documento (protección memoria)
MAX_CHUNKS_PER_DOCUMENT = 500