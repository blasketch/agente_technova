"""
Motor de NLP para el agente de soporte técnico de TechNova Gadgets S.L.
Implementa búsqueda semántica usando embeddings de Sentence Transformers.
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
from typing import List, Dict, Tuple
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TechNovaNLPEngine:
    """
    Motor de procesamiento de lenguaje natural para el agente de soporte técnico.
    Utiliza embeddings semánticos para encontrar las respuestas más relevantes.
    """
    
    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        """
        Inicializa el motor de NLP.
        
        Args:
            model_name: Nombre del modelo de Sentence Transformers a usar
        """
        self.model_name = model_name
        self.model = None
        self.faq_data = None
        self.question_embeddings = None
        self.embeddings_file = "data/question_embeddings.pkl"
        self.faq_file = "data/faq.csv"
        
        # Cargar datos y modelo
        self._load_model()
        self._load_faq_data()
        self._load_or_generate_embeddings()
    
    def _load_model(self):
        """Carga el modelo de Sentence Transformers."""
        try:
            logger.info(f"Cargando modelo: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("Modelo cargado exitosamente")
        except Exception as e:
            logger.error(f"Error cargando el modelo: {e}")
            raise
    
    def _load_faq_data(self):
        """Carga los datos de FAQ desde el archivo CSV."""
        try:
            if not os.path.exists(self.faq_file):
                raise FileNotFoundError(f"Archivo FAQ no encontrado: {self.faq_file}")
            
            self.faq_data = pd.read_csv(self.faq_file)
            logger.info(f"FAQ cargado: {len(self.faq_data)} preguntas")
            
            # Validar estructura
            required_columns = ['pregunta', 'respuesta', 'categoria']
            if not all(col in self.faq_data.columns for col in required_columns):
                raise ValueError(f"El archivo FAQ debe contener las columnas: {required_columns}")
                
        except Exception as e:
            logger.error(f"Error cargando FAQ: {e}")
            raise
    
    def _load_or_generate_embeddings(self):
        """Carga embeddings existentes o los genera si no existen."""
        try:
            if os.path.exists(self.embeddings_file):
                logger.info("Cargando embeddings existentes...")
                with open(self.embeddings_file, 'rb') as f:
                    self.question_embeddings = pickle.load(f)
                logger.info("Embeddings cargados exitosamente")
            else:
                logger.info("Generando nuevos embeddings...")
                self._generate_embeddings()
                
        except Exception as e:
            logger.error(f"Error con embeddings: {e}")
            # Si hay error, regenerar embeddings
            self._generate_embeddings()
    
    def _generate_embeddings(self):
        """Genera embeddings para todas las preguntas del FAQ."""
        try:
            logger.info("Generando embeddings para preguntas...")
            questions = self.faq_data['pregunta'].tolist()
            
            # Generar embeddings
            self.question_embeddings = self.model.encode(
                questions, 
                show_progress_bar=True,
                convert_to_numpy=True
            )
            
            # Guardar embeddings
            os.makedirs(os.path.dirname(self.embeddings_file), exist_ok=True)
            with open(self.embeddings_file, 'wb') as f:
                pickle.dump(self.question_embeddings, f)
            
            logger.info(f"Embeddings generados y guardados: {self.question_embeddings.shape}")
            
        except Exception as e:
            logger.error(f"Error generando embeddings: {e}")
            raise
    
    def search_similar_questions(self, query: str, top_k: int = 3, threshold: float = 0.3) -> List[Dict]:
        """
        Busca preguntas similares usando similitud coseno.
        
        Args:
            query: Consulta del usuario
            top_k: Número de resultados a retornar
            threshold: Umbral mínimo de similitud
            
        Returns:
            Lista de diccionarios con preguntas similares y sus scores
        """
        try:
            # Generar embedding para la consulta
            query_embedding = self.model.encode([query], convert_to_numpy=True)
            
            # Calcular similitudes
            similarities = cosine_similarity(query_embedding, self.question_embeddings)[0]
            
            # Obtener índices ordenados por similitud
            sorted_indices = np.argsort(similarities)[::-1]
            
            # Filtrar por umbral y obtener top_k
            results = []
            for idx in sorted_indices:
                if similarities[idx] >= threshold and len(results) < top_k:
                    result = {
                        'index': idx,
                        'question': self.faq_data.iloc[idx]['pregunta'],
                        'answer': self.faq_data.iloc[idx]['respuesta'],
                        'category': self.faq_data.iloc[idx]['categoria'],
                        'similarity_score': float(similarities[idx])
                    }
                    results.append(result)
            
            logger.info(f"Búsqueda completada: {len(results)} resultados para '{query}'")
            return results
            
        except Exception as e:
            logger.error(f"Error en búsqueda: {e}")
            return []
    
    def get_best_answer(self, query: str, threshold: float = 0.3) -> Dict:
        """
        Obtiene la mejor respuesta para una consulta.
        
        Args:
            query: Consulta del usuario
            threshold: Umbral mínimo de similitud
            
        Returns:
            Diccionario con la mejor respuesta o indicación de no encontrado
        """
        results = self.search_similar_questions(query, top_k=1, threshold=threshold)
        
        if results:
            best_result = results[0]
            return {
                'found': True,
                'answer': best_result['answer'],
                'question': best_result['question'],
                'category': best_result['category'],
                'confidence': best_result['similarity_score'],
                'suggestion': f"Basado en: '{best_result['question']}'"
            }
        else:
            return {
                'found': False,
                'answer': "Lo siento, no encontré una respuesta específica para tu consulta. Te recomiendo contactar con nuestro soporte técnico al 900-TECHNOVA o enviar un email a soporte@technova.com",
                'confidence': 0.0,
                'suggestion': "Contacta con soporte técnico para asistencia personalizada"
            }
    
    def get_category_stats(self) -> Dict:
        """Retorna estadísticas por categoría del FAQ."""
        try:
            stats = self.faq_data['categoria'].value_counts().to_dict()
            return {
                'total_questions': len(self.faq_data),
                'categories': stats,
                'categories_count': len(stats)
            }
        except Exception as e:
            logger.error(f"Error obteniendo estadísticas: {e}")
            return {}
    
    def add_new_qa(self, question: str, answer: str, category: str):
        """
        Añade una nueva pregunta-respuesta al FAQ.
        Nota: Requiere regenerar embeddings después de añadir.
        
        Args:
            question: Nueva pregunta
            answer: Nueva respuesta
            category: Categoría de la pregunta
        """
        try:
            # Añadir al DataFrame
            new_row = pd.DataFrame({
                'pregunta': [question],
                'respuesta': [answer],
                'categoria': [category]
            })
            
            self.faq_data = pd.concat([self.faq_data, new_row], ignore_index=True)
            
            # Guardar FAQ actualizado
            self.faq_data.to_csv(self.faq_file, index=False)
            
            logger.info(f"Nueva Q&A añadida: '{question[:50]}...'")
            
            # Regenerar embeddings
            self._generate_embeddings()
            
        except Exception as e:
            logger.error(f"Error añadiendo nueva Q&A: {e}")
            raise


def create_nlp_engine() -> TechNovaNLPEngine:
    """
    Función de conveniencia para crear una instancia del motor NLP.
    
    Returns:
        Instancia de TechNovaNLPEngine
    """
    return TechNovaNLPEngine()


if __name__ == "__main__":
    # Prueba básica del motor
    print("🤖 Inicializando motor NLP de TechNova...")
    
    try:
        engine = create_nlp_engine()
        
        # Prueba de búsqueda
        test_queries = [
            "Mi auricular no funciona",
            "¿Cómo cambio la correa del reloj?",
            "Batería se agota rápido",
            "No puedo conectar por Bluetooth"
        ]
        
        print("\n🔍 Pruebas de búsqueda:")
        for query in test_queries:
            print(f"\nConsulta: '{query}'")
            result = engine.get_best_answer(query)
            print(f"Encontrado: {result['found']}")
            print(f"Confianza: {result['confidence']:.3f}")
            print(f"Respuesta: {result['answer'][:100]}...")
        
        # Estadísticas
        print(f"\n📊 Estadísticas del FAQ:")
        stats = engine.get_category_stats()
        print(f"Total preguntas: {stats['total_questions']}")
        print(f"Categorías: {stats['categories']}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
