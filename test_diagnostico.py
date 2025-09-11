#!/usr/bin/env python3
"""
Script de prueba para diagnosticar problemas con la API
"""

import sys
import os

# Añadir el directorio actual al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Probar imports de módulos"""
    print("🔍 Probando imports...")
    
    try:
        from app.nlp_engine import TechNovaNLPEngine, create_nlp_engine
        print("✅ nlp_engine importado correctamente")
    except Exception as e:
        print(f"❌ Error importando nlp_engine: {e}")
        return False
    
    try:
        from app.utils import validate_query, categorize_query, extract_keywords
        print("✅ utils importado correctamente")
    except Exception as e:
        print(f"❌ Error importando utils: {e}")
        return False
    
    return True

def test_nlp_engine():
    """Probar motor NLP"""
    print("\n🤖 Probando motor NLP...")
    
    try:
        from app.nlp_engine import create_nlp_engine
        engine = create_nlp_engine()
        print("✅ Motor NLP creado correctamente")
        
        # Probar búsqueda
        result = engine.get_best_answer("Mi auricular no funciona")
        print(f"✅ Búsqueda exitosa: {result['found']}")
        print(f"📝 Respuesta: {result['answer'][:50]}...")
        
        return True
    except Exception as e:
        print(f"❌ Error con motor NLP: {e}")
        return False

def test_api_creation():
    """Probar creación de API"""
    print("\n🌐 Probando creación de API...")
    
    try:
        from app.main import app
        print("✅ API creada correctamente")
        
        # Probar cliente de prueba
        from fastapi.testclient import TestClient
        client = TestClient(app)
        
        # Probar endpoint raíz
        response = client.get("/")
        print(f"✅ Endpoint raíz: {response.status_code}")
        
        # Probar endpoint de salud
        response = client.get("/health")
        print(f"✅ Endpoint salud: {response.status_code}")
        
        return True
    except Exception as e:
        print(f"❌ Error con API: {e}")
        return False

def test_api_endpoint():
    """Probar endpoint de soporte"""
    print("\n🔧 Probando endpoint de soporte...")
    
    try:
        from app.main import app
        from fastapi.testclient import TestClient
        
        client = TestClient(app)
        
        # Probar endpoint de soporte
        data = {
            "query": "Mi auricular no funciona",
            "threshold": 0.3
        }
        
        response = client.post("/soporte", json=data)
        print(f"✅ Endpoint soporte: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"📝 Respuesta: {result['answer'][:50]}...")
            print(f"🎯 Confianza: {result['confidence']}")
        else:
            print(f"❌ Error: {response.text}")
        
        return response.status_code == 200
        
    except Exception as e:
        print(f"❌ Error probando endpoint: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Diagnóstico del Agente TechNova")
    print("=" * 50)
    
    # Ejecutar pruebas
    tests = [
        ("Imports", test_imports),
        ("Motor NLP", test_nlp_engine),
        ("Creación API", test_api_creation),
        ("Endpoint Soporte", test_api_endpoint)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ Error en {name}: {e}")
            results.append((name, False))
    
    # Resumen
    print("\n📊 Resumen de Pruebas:")
    print("=" * 30)
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name}: {status}")
    
    # Resultado final
    all_passed = all(result for _, result in results)
    if all_passed:
        print("\n🎉 ¡Todas las pruebas pasaron!")
    else:
        print("\n⚠️ Algunas pruebas fallaron. Revisa los errores arriba.")
