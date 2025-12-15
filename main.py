import logging
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from api.endpoints import register_routes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Get Talent RAG API",
    description="Sistema RAG para búsqueda semántica y generación de respuestas",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

register_routes(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Incoming request: {request.method} {request.url.path}")
    try:
        response = await call_next(request)
        logger.info(f"Response status: {response.status_code}")
        return response
    except Exception as e:
        logger.error(f"Unhandled exception: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": "Error interno del servidor"},
        )

@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"error": "Recurso no encontrado"},
    )

@app.get("/", tags=["Health"])
async def root():
    return {
        "message": "Get Talent RAG API está funcionando",
        "version": "1.0.0",
        "endpoints": {
            "upload": "POST /upload",
            "generate_embeddings": "POST /generate-embeddings",
            "search": "POST /search",
            "ask": "POST /ask",
        },
    }

@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "healthy", "service": "rag-api"}