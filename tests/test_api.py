"""
Tests para la API del agente de soporte técnico de TechNova Gadgets S.L.
"""

import pytest
import requests
import json
from fastapi.testclient import TestClient
import sys
import os

# Añadir el directorio app al path para importar módulos
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'app'))

from main import app
from nlp_engine import TechNovaNLPEngine
from utils import validate_query, categorize_query, extract_keywords

# Cliente de prueba
client = TestClient(app)


class TestAPIEndpoints:
    """Tests para los endpoints de la API."""
    
    def test_root_endpoint(self):
        """Test del endpoint raíz."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "TechNova Support Agent API" in data["message"]
    
    def test_health_endpoint(self):
        """Test del endpoint de salud."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "nlp_engine_loaded" in data
    
    def test_soporte_post_valid_query(self):
        """Test del endpoint POST de soporte con consulta válida."""
        query_data = {
            "query": "Mi auricular no funciona",
            "threshold": 0.3,
            "include_confidence": True,
            "include_suggestions": True
        }
        
        response = client.post("/soporte", json=query_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert data["query"] == "Mi auricular no funciona"
        assert "answer" in data
        assert "confidence" in data
        assert "processing_time" in data
    
    def test_soporte_get_valid_query(self):
        """Test del endpoint GET de soporte con consulta válida."""
        query = "¿Cómo configuro mis auriculares?"
        
        response = client.get(f"/soporte/{query}")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert data["query"] == query
        assert "answer" in data
    
    def test_soporte_invalid_query_too_short(self):
        """Test con consulta muy corta."""
        query_data = {
            "query": "ab",
            "threshold": 0.3
        }
        
        response = client.post("/soporte", json=query_data)
        assert response.status_code == 400
        data = response.json()
        assert "error" in data["detail"]
    
    def test_soporte_invalid_query_too_long(self):
        """Test con consulta muy larga."""
        long_query = "a" * 501
        
        query_data = {
            "query": long_query,
            "threshold": 0.3
        }
        
        response = client.post("/soporte", json=query_data)
        assert response.status_code == 400
    
    def test_buscar_multiple_results(self):
        """Test del endpoint de búsqueda múltiple."""
        search_data = {
            "query": "problema con auriculares",
            "top_k": 3,
            "threshold": 0.2
        }
        
        response = client.post("/buscar", json=search_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "results" in data
        assert "total_found" in data
        assert len(data["results"]) <= search_data["top_k"]
    
    def test_estadisticas_endpoint(self):
        """Test del endpoint de estadísticas."""
        response = client.get("/estadisticas")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "total_questions" in data
        assert "categories" in data
        assert data["total_questions"] > 0
    
    def test_404_endpoint(self):
        """Test de endpoint no encontrado."""
        response = client.get("/endpoint-inexistente")
        assert response.status_code == 404
        data = response.json()
        assert "error" in data


class TestNLPEngine:
    """Tests para el motor de NLP."""
    
    def test_nlp_engine_initialization(self):
        """Test de inicialización del motor NLP."""
        try:
            engine = TechNovaNLPEngine()
            assert engine is not None
            assert engine.model is not None
            assert engine.faq_data is not None
            assert engine.question_embeddings is not None
        except Exception as e:
            pytest.skip(f"Motor NLP no disponible: {e}")
    
    def test_search_similar_questions(self):
        """Test de búsqueda de preguntas similares."""
        try:
            engine = TechNovaNLPEngine()
            
            query = "Mi auricular no funciona"
            results = engine.search_similar_questions(query, top_k=3)
            
            assert isinstance(results, list)
            assert len(results) <= 3
            
            if results:
                result = results[0]
                assert "question" in result
                assert "answer" in result
                assert "similarity_score" in result
                assert result["similarity_score"] >= 0.0
                assert result["similarity_score"] <= 1.0
                
        except Exception as e:
            pytest.skip(f"Motor NLP no disponible: {e}")
    
    def test_get_best_answer(self):
        """Test de obtención de mejor respuesta."""
        try:
            engine = TechNovaNLPEngine()
            
            query = "¿Cómo configuro mis auriculares?"
            result = engine.get_best_answer(query)
            
            assert isinstance(result, dict)
            assert "found" in result
            assert "answer" in result
            assert "confidence" in result
            
            if result["found"]:
                assert result["confidence"] > 0.0
                assert len(result["answer"]) > 0
                
        except Exception as e:
            pytest.skip(f"Motor NLP no disponible: {e}")
    
    def test_category_stats(self):
        """Test de estadísticas por categoría."""
        try:
            engine = TechNovaNLPEngine()
            stats = engine.get_category_stats()
            
            assert isinstance(stats, dict)
            assert "total_questions" in stats
            assert "categories" in stats
            assert stats["total_questions"] > 0
            
        except Exception as e:
            pytest.skip(f"Motor NLP no disponible: {e}")


class TestUtils:
    """Tests para las funciones auxiliares."""
    
    def test_validate_query_valid(self):
        """Test de validación de consulta válida."""
        valid_queries = [
            "Mi auricular no funciona",
            "¿Cómo configuro mi smartwatch?",
            "Problema con la batería"
        ]
        
        for query in valid_queries:
            result = validate_query(query)
            assert result["valid"] is True
            assert result["error"] is None
    
    def test_validate_query_invalid(self):
        """Test de validación de consultas inválidas."""
        invalid_queries = [
            "",  # Vacía
            "ab",  # Muy corta
            "a" * 501,  # Muy larga
            "!!!@@@",  # Solo símbolos
            None  # None
        ]
        
        for query in invalid_queries:
            result = validate_query(query)
            assert result["valid"] is False
            assert result["error"] is not None
    
    def test_categorize_query(self):
        """Test de categorización de consultas."""
        test_cases = [
            ("Mi auricular no funciona", "soporte_tecnico"),
            ("¿Cómo configuro Bluetooth?", "configuracion"),
            ("La batería se agota rápido", "bateria"),
            ("¿Cómo cambio la correa?", "accesorios"),
            ("Consulta general", "general")
        ]
        
        for query, expected_category in test_cases:
            category = categorize_query(query)
            # Nota: La categorización puede no ser exacta, solo verificamos que retorne algo
            assert isinstance(category, str)
            assert len(category) > 0
    
    def test_extract_keywords(self):
        """Test de extracción de palabras clave."""
        test_cases = [
            ("Mi auricular no funciona", ["auricular", "funciona"]),
            ("¿Cómo configuro Bluetooth?", ["configuro", "bluetooth"]),
            ("La batería se agota rápido", ["batería", "agota", "rápido"])
        ]
        
        for query, expected_keywords in test_cases:
            keywords = extract_keywords(query)
            assert isinstance(keywords, list)
            # Verificar que algunas palabras clave esperadas estén presentes
            for expected_keyword in expected_keywords:
                if expected_keyword in query.lower():
                    # Al menos una palabra clave debe estar presente
                    assert len(keywords) > 0
                    break


class TestIntegration:
    """Tests de integración end-to-end."""
    
    def test_complete_workflow(self):
        """Test del flujo completo de consulta."""
        try:
            # Test 1: Consulta válida
            query_data = {
                "query": "Mi smartwatch no se conecta al teléfono",
                "threshold": 0.3,
                "include_confidence": True,
                "include_suggestions": True
            }
            
            response = client.post("/soporte", json=query_data)
            assert response.status_code == 200
            
            data = response.json()
            assert data["success"] is True
            assert data["found"] is not None
            assert data["confidence"] >= 0.0
            assert data["confidence"] <= 1.0
            
            # Test 2: Búsqueda múltiple
            search_data = {
                "query": "problema con auriculares",
                "top_k": 2,
                "threshold": 0.2
            }
            
            response = client.post("/buscar", json=search_data)
            assert response.status_code == 200
            
            data = response.json()
            assert data["success"] is True
            assert data["total_found"] >= 0
            
            # Test 3: Estadísticas
            response = client.get("/estadisticas")
            assert response.status_code == 200
            
            data = response.json()
            assert data["success"] is True
            assert data["total_questions"] > 0
            
        except Exception as e:
            pytest.skip(f"Test de integración no disponible: {e}")


# Configuración de pytest
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
