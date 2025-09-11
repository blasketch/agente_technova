"""
Funciones auxiliares para el agente de soporte técnico de TechNova Gadgets S.L.
"""

import re
import string
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """
    Limpia y normaliza el texto de entrada.
    
    Args:
        text: Texto a limpiar
        
    Returns:
        Texto limpio y normalizado
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Convertir a minúsculas
    text = text.lower().strip()
    
    # Remover caracteres especiales pero mantener acentos y ñ
    text = re.sub(r'[^\w\sáéíóúñü]', ' ', text)
    
    # Remover espacios múltiples
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def extract_keywords(text: str, min_length: int = 3) -> List[str]:
    """
    Extrae palabras clave relevantes del texto.
    
    Args:
        text: Texto del cual extraer palabras clave
        min_length: Longitud mínima de las palabras
        
    Returns:
        Lista de palabras clave
    """
    if not text:
        return []
    
    # Limpiar texto
    clean_text_input = clean_text(text)
    
    # Palabras comunes a ignorar
    stop_words = {
        'que', 'como', 'para', 'con', 'por', 'del', 'las', 'los', 'una', 'uno',
        'este', 'esta', 'estos', 'estas', 'mi', 'mis', 'tu', 'tus', 'su', 'sus',
        'nos', 'nosotros', 'ellos', 'ellas', 'me', 'te', 'se', 'le', 'lo', 'la',
        'el', 'de', 'en', 'a', 'y', 'o', 'pero', 'si', 'no', 'muy', 'mas', 'menos',
        'hacer', 'tener', 'ser', 'estar', 'haber', 'poder', 'querer', 'decir',
        'ver', 'saber', 'ir', 'venir', 'dar', 'tomar', 'poner', 'salir', 'entrar'
    }
    
    # Dividir en palabras y filtrar
    words = clean_text_input.split()
    keywords = [
        word for word in words 
        if len(word) >= min_length and word not in stop_words
    ]
    
    return keywords


def categorize_query(query: str) -> str:
    """
    Categoriza una consulta basándose en palabras clave.
    
    Args:
        query: Consulta del usuario
        
    Returns:
        Categoría estimada
    """
    if not query:
        return "general"
    
    query_lower = query.lower()
    
    # Palabras clave por categoría
    categories = {
        'configuracion': ['configurar', 'configuración', 'instalar', 'instalación', 'setup', 'config'],
        'conectividad': ['bluetooth', 'wifi', 'conectar', 'conexión', 'emparejar', 'sincronizar'],
        'bateria': ['batería', 'carga', 'cargar', 'energía', 'duración', 'agotar'],
        'soporte_tecnico': ['problema', 'error', 'fallo', 'no funciona', 'roto', 'dañado'],
        'mantenimiento': ['limpiar', 'limpieza', 'mantener', 'cuidado', 'conservar'],
        'garantia': ['garantía', 'devolución', 'reembolso', 'cambio', 'reparar'],
        'accesorios': ['correa', 'cable', 'cargador', 'estuche', 'filtro'],
        'funciones': ['función', 'característica', 'modo', 'configuración', 'ajuste'],
        'llamadas': ['llamada', 'teléfono', 'micrófono', 'audio', 'voz'],
        'actualizaciones': ['actualizar', 'firmware', 'software', 'versión', 'update']
    }
    
    # Contar coincidencias por categoría
    category_scores = {}
    for category, keywords in categories.items():
        score = sum(1 for keyword in keywords if keyword in query_lower)
        if score > 0:
            category_scores[category] = score
    
    # Retornar la categoría con mayor score
    if category_scores:
        return max(category_scores, key=category_scores.get)
    
    return "general"


def format_response(response_data: Dict, include_confidence: bool = True) -> str:
    """
    Formatea la respuesta del agente para presentación al usuario.
    
    Args:
        response_data: Datos de la respuesta del motor NLP
        include_confidence: Si incluir el nivel de confianza
        
    Returns:
        Respuesta formateada
    """
    if not response_data:
        return "Lo siento, no pude procesar tu consulta."
    
    # Construir respuesta
    answer_parts = []
    
    # Respuesta principal
    answer_parts.append(response_data.get('answer', ''))
    
    # Información adicional si está disponible
    if response_data.get('found', False):
        if 'suggestion' in response_data:
            answer_parts.append(f"\n💡 {response_data['suggestion']}")
        
        if include_confidence and 'confidence' in response_data:
            confidence = response_data['confidence']
            if confidence > 0.8:
                confidence_text = "Alta confianza"
            elif confidence > 0.5:
                confidence_text = "Confianza media"
            else:
                confidence_text = "Confianza baja"
            answer_parts.append(f"\n🎯 {confidence_text} ({confidence:.1%})")
    
    return "\n".join(answer_parts)


def validate_query(query: str) -> Dict[str, any]:
    """
    Valida una consulta del usuario.
    
    Args:
        query: Consulta a validar
        
    Returns:
        Diccionario con resultado de validación
    """
    result = {
        'valid': True,
        'error': None,
        'suggestions': []
    }
    
    if not query or not isinstance(query, str):
        result['valid'] = False
        result['error'] = "La consulta no puede estar vacía"
        return result
    
    # Limpiar y verificar longitud
    clean_query = clean_text(query)
    
    if len(clean_query) < 3:
        result['valid'] = False
        result['error'] = "La consulta es demasiado corta"
        result['suggestions'].append("Intenta ser más específico en tu pregunta")
        return result
    
    if len(clean_query) > 500:
        result['valid'] = False
        result['error'] = "La consulta es demasiado larga"
        result['suggestions'].append("Intenta hacer una pregunta más concisa")
        return result
    
    # Verificar si contiene solo caracteres especiales
    if not re.search(r'[a-záéíóúñü]', clean_query):
        result['valid'] = False
        result['error'] = "La consulta debe contener texto válido"
        result['suggestions'].append("Usa palabras en lugar de solo símbolos")
        return result
    
    return result


def generate_fallback_response(query: str) -> str:
    """
    Genera una respuesta de respaldo cuando no se encuentra una respuesta específica.
    
    Args:
        query: Consulta original del usuario
        
    Returns:
        Respuesta de respaldo personalizada
    """
    # Categorizar la consulta para personalizar la respuesta
    category = categorize_query(query)
    
    fallback_responses = {
        'configuracion': "Para problemas de configuración, te recomiendo revisar el manual de usuario o contactar con nuestro soporte técnico especializado.",
        'conectividad': "Si tienes problemas de conectividad, verifica que ambos dispositivos tengan Bluetooth/WiFi activado y estén cerca uno del otro.",
        'bateria': "Para problemas de batería, asegúrate de usar el cargador original y verifica que los contactos estén limpios.",
        'soporte_tecnico': "Para asistencia técnica inmediata, contacta con nuestro equipo de soporte al 900-TECHNOVA.",
        'garantia': "Para consultas sobre garantía y devoluciones, puedes contactar con nuestro servicio al cliente.",
        'general': "Nuestro equipo de soporte técnico está disponible 24/7 para ayudarte. Contacta con nosotros para asistencia personalizada."
    }
    
    base_response = fallback_responses.get(category, fallback_responses['general'])
    
    return f"{base_response}\n\n📞 Teléfono: 900-TECHNOVA\n📧 Email: soporte@technova.com\n💬 Chat en vivo disponible en nuestra web"


def extract_product_mentions(query: str) -> List[str]:
    """
    Extrae menciones de productos específicos en la consulta.
    
    Args:
        query: Consulta del usuario
        
    Returns:
        Lista de productos mencionados
    """
    if not query:
        return []
    
    # Productos comunes de TechNova
    products = {
        'auriculares': ['auricular', 'auriculares', 'headphones', 'earbuds'],
        'smartwatch': ['smartwatch', 'reloj', 'watch', 'reloj inteligente'],
        'cargador': ['cargador', 'cable', 'cable de carga'],
        'correa': ['correa', 'banda', 'pulsera'],
        'estuche': ['estuche', 'case', 'funda']
    }
    
    query_lower = query.lower()
    mentioned_products = []
    
    for product, keywords in products.items():
        if any(keyword in query_lower for keyword in keywords):
            mentioned_products.append(product)
    
    return mentioned_products


def create_contextual_response(query: str, response_data: Dict) -> str:
    """
    Crea una respuesta contextual basada en la consulta y los datos de respuesta.
    
    Args:
        query: Consulta original
        response_data: Datos de respuesta del motor NLP
        
    Returns:
        Respuesta contextual mejorada
    """
    # Obtener productos mencionados
    products = extract_product_mentions(query)
    
    # Respuesta base
    base_response = format_response(response_data)
    
    # Añadir contexto específico del producto si es relevante
    if products and response_data.get('found', False):
        product_context = {
            'auriculares': "💡 Consejo adicional: Si el problema persiste, prueba reiniciar los auriculares manteniendo presionado el botón de encendido por 10 segundos.",
            'smartwatch': "💡 Consejo adicional: Para problemas de sincronización, asegúrate de que la app TechNova Health esté actualizada.",
            'cargador': "💡 Consejo adicional: Usa siempre el cargador original para evitar daños en la batería."
        }
        
        for product in products:
            if product in product_context:
                base_response += f"\n\n{product_context[product]}"
                break
    
    return base_response


if __name__ == "__main__":
    # Pruebas de las funciones auxiliares
    print("🧪 Probando funciones auxiliares...")
    
    test_queries = [
        "Mi auricular no funciona bien",
        "¿Cómo cambio la correa del smartwatch?",
        "La batería se agota muy rápido",
        "No puedo conectar por Bluetooth"
    ]
    
    for query in test_queries:
        print(f"\nConsulta: '{query}'")
        
        # Validar
        validation = validate_query(query)
        print(f"Válida: {validation['valid']}")
        
        # Categorizar
        category = categorize_query(query)
        print(f"Categoría: {category}")
        
        # Extraer palabras clave
        keywords = extract_keywords(query)
        print(f"Palabras clave: {keywords}")
        
        # Extraer productos
        products = extract_product_mentions(query)
        print(f"Productos: {products}")
        
        print("-" * 50)
