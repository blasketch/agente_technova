"""
API REST para el agente de soporte técnico de TechNova Gadgets S.L.
Implementa endpoints para consultas de soporte técnico usando FastAPI.
"""

from fastapi import FastAPI, HTTPException, Query, Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import logging
import time
from datetime import datetime

# Configurar logging primero
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Importar módulos locales
try:
    # Intentar imports relativos primero
    try:
        from .nlp_engine import TechNovaNLPEngine, create_nlp_engine
        from .utils import (
            validate_query, 
            categorize_query, 
            extract_keywords, 
            create_contextual_response,
            generate_fallback_response,
            extract_product_mentions
        )
    except ImportError:
        # Si fallan los imports relativos, intentar imports absolutos
        from nlp_engine import TechNovaNLPEngine, create_nlp_engine
        from utils import (
            validate_query, 
            categorize_query, 
            extract_keywords, 
            create_contextual_response,
            generate_fallback_response,
            extract_product_mentions
        )
except ImportError as e:
    logger.error(f"Error importando módulos locales: {e}")
    # Crear funciones dummy para evitar errores
    def create_nlp_engine():
        raise Exception("Motor NLP no disponible")
    
    def validate_query(query):
        return {"valid": False, "error": "Motor NLP no disponible"}
    
    def categorize_query(query):
        return "general"
    
    def extract_keywords(query):
        return []
    
    def create_contextual_response(query, response_data):
        return response_data.get('answer', 'Error')
    
    def generate_fallback_response(query):
        return "Motor NLP no disponible"
    
    def extract_product_mentions(query):
        return []

# Crear aplicación FastAPI
app = FastAPI(
    title="TechNova Support Agent API",
    description="API del agente de soporte técnico para TechNova Gadgets S.L.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especificar dominios específicos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar motor NLP (singleton)
nlp_engine = None

def get_nlp_engine():
    """Obtiene la instancia del motor NLP (lazy loading)."""
    global nlp_engine
    if nlp_engine is None:
        try:
            logger.info("Inicializando motor NLP...")
            nlp_engine = create_nlp_engine()
            logger.info("Motor NLP inicializado")
        except Exception as e:
            logger.error(f"Error inicializando motor NLP: {e}")
            raise HTTPException(status_code=500, detail=f"Error inicializando motor NLP: {str(e)}")
    return nlp_engine


# Modelos Pydantic para validación de datos
class QueryRequest(BaseModel):
    """Modelo para solicitudes de consulta."""
    query: str = Field(..., min_length=3, max_length=500, description="Consulta del usuario")
    threshold: Optional[float] = Field(0.3, ge=0.0, le=1.0, description="Umbral de similitud mínimo")
    include_confidence: Optional[bool] = Field(True, description="Incluir nivel de confianza")
    include_suggestions: Optional[bool] = Field(True, description="Incluir sugerencias adicionales")


class QueryResponse(BaseModel):
    """Modelo para respuestas de consulta."""
    success: bool
    query: str
    answer: str
    found: bool
    confidence: float
    category: Optional[str] = None
    suggestion: Optional[str] = None
    keywords: Optional[List[str]] = None
    products_mentioned: Optional[List[str]] = None
    processing_time: float
    timestamp: str


class SearchRequest(BaseModel):
    """Modelo para solicitudes de búsqueda múltiple."""
    query: str = Field(..., min_length=3, max_length=500)
    top_k: Optional[int] = Field(3, ge=1, le=10)
    threshold: Optional[float] = Field(0.3, ge=0.0, le=1.0)


class SearchResult(BaseModel):
    """Modelo para resultados de búsqueda."""
    question: str
    answer: str
    category: str
    similarity_score: float


class SearchResponse(BaseModel):
    """Modelo para respuestas de búsqueda múltiple."""
    success: bool
    query: str
    results: List[SearchResult]
    total_found: int
    processing_time: float
    timestamp: str


class StatsResponse(BaseModel):
    """Modelo para estadísticas del sistema."""
    success: bool
    total_questions: int
    categories: Dict[str, int]
    categories_count: int
    timestamp: str


class HealthResponse(BaseModel):
    """Modelo para estado de salud de la API."""
    status: str
    version: str
    nlp_engine_loaded: bool
    timestamp: str


# Endpoints de la API

@app.get("/", response_model=Dict[str, str])
async def root():
    """Endpoint raíz con información básica."""
    return {
        "message": "🤖 TechNova Support Agent API",
        "description": "API del agente de soporte técnico para TechNova Gadgets S.L.",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Verifica el estado de salud de la API."""
    try:
        engine = get_nlp_engine()
        nlp_loaded = engine is not None
    except Exception as e:
        logger.error(f"Error verificando motor NLP: {e}")
        nlp_loaded = False
    
    return HealthResponse(
        status="healthy" if nlp_loaded else "degraded",
        version="1.0.0",
        nlp_engine_loaded=nlp_loaded,
        timestamp=datetime.now().isoformat()
    )


@app.post("/soporte", response_model=QueryResponse)
async def consultar_soporte(request: QueryRequest):
    """
    Endpoint principal para consultas de soporte técnico.
    
    Procesa una consulta del usuario y retorna la mejor respuesta encontrada
    en la base de conocimiento de TechNova.
    """
    start_time = time.time()
    
    try:
        # Validar consulta
        validation = validate_query(request.query)
        if not validation['valid']:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": validation['error'],
                    "suggestions": validation['suggestions']
                }
            )
        
        # Obtener motor NLP
        engine = get_nlp_engine()
        
        # Procesar consulta
        result = engine.get_best_answer(request.query, request.threshold)
        
        # Extraer información adicional
        category = categorize_query(request.query)
        keywords = extract_keywords(request.query)
        products = extract_product_mentions(request.query)
        
        # Crear respuesta contextual
        if request.include_suggestions and result['found']:
            answer = create_contextual_response(request.query, result)
        else:
            answer = result['answer']
        
        processing_time = time.time() - start_time
        
        return QueryResponse(
            success=True,
            query=request.query,
            answer=answer,
            found=result['found'],
            confidence=result['confidence'],
            category=category,
            suggestion=result.get('suggestion') if request.include_suggestions else None,
            keywords=keywords if request.include_suggestions else None,
            products_mentioned=products if request.include_suggestions else None,
            processing_time=round(processing_time, 3),
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error procesando consulta: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")


@app.post("/buscar", response_model=SearchResponse)
async def buscar_multiple(request: SearchRequest):
    """
    Endpoint para búsqueda múltiple de preguntas similares.
    
    Retorna múltiples resultados ordenados por similitud.
    """
    start_time = time.time()
    
    try:
        # Validar consulta
        validation = validate_query(request.query)
        if not validation['valid']:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": validation['error'],
                    "suggestions": validation['suggestions']
                }
            )
        
        # Obtener motor NLP
        engine = get_nlp_engine()
        
        # Buscar resultados similares
        results = engine.search_similar_questions(
            request.query, 
            request.top_k, 
            request.threshold
        )
        
        # Convertir a modelo de respuesta
        search_results = [
            SearchResult(
                question=r['question'],
                answer=r['answer'],
                category=r['category'],
                similarity_score=r['similarity_score']
            )
            for r in results
        ]
        
        processing_time = time.time() - start_time
        
        return SearchResponse(
            success=True,
            query=request.query,
            results=search_results,
            total_found=len(search_results),
            processing_time=round(processing_time, 3),
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en búsqueda múltiple: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")


@app.get("/estadisticas", response_model=StatsResponse)
async def obtener_estadisticas():
    """Obtiene estadísticas del sistema y base de conocimiento."""
    try:
        engine = get_nlp_engine()
        stats = engine.get_category_stats()
        
        return StatsResponse(
            success=True,
            total_questions=stats['total_questions'],
            categories=stats['categories'],
            categories_count=stats['categories_count'],
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error obteniendo estadísticas: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")


@app.get("/soporte/{query}", response_model=QueryResponse)
async def consultar_soporte_get(
    query: str = Path(..., min_length=3, max_length=500),
    threshold: float = Query(0.3, ge=0.0, le=1.0),
    include_confidence: bool = Query(True),
    include_suggestions: bool = Query(True)
):
    """
    Endpoint GET para consultas de soporte técnico.
    
    Versión simplificada del endpoint POST para consultas rápidas.
    """
    # Crear request object
    request = QueryRequest(
        query=query,
        threshold=threshold,
        include_confidence=include_confidence,
        include_suggestions=include_suggestions
    )
    
    # Reutilizar lógica del endpoint POST
    return await consultar_soporte(request)


# Manejo de errores personalizado
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "error": "Endpoint no encontrado",
            "message": "Verifica la URL y consulta la documentación en /docs",
            "available_endpoints": [
                "/",
                "/health",
                "/soporte",
                "/buscar",
                "/estadisticas",
                "/docs"
            ]
        }
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Error interno: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Error interno del servidor",
            "message": "Contacta con el administrador del sistema",
            "support": "soporte@technova.com"
        }
    )


# Eventos de la aplicación
@app.on_event("startup")
async def startup_event():
    """Evento de inicio de la aplicación."""
    logger.info("🚀 Iniciando TechNova Support Agent API...")
    try:
        # Precargar motor NLP
        get_nlp_engine()
        logger.info("✅ API iniciada correctamente")
    except Exception as e:
        logger.error(f"❌ Error iniciando API: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Evento de cierre de la aplicación."""
    logger.info("🛑 Cerrando TechNova Support Agent API...")


if __name__ == "__main__":
    import uvicorn
    
    # Configuración para desarrollo
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
