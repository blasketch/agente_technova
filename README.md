🤖 Agente de Soporte Técnico – TechNova Gadgets S.L.

Este proyecto implementa un agente de inteligencia artificial capaz de asistir al soporte técnico de una empresa ficticia dedicada a la venta de artículos de tecnología y gadgets: TechNova Gadgets S.L.

El objetivo es demostrar cómo la IA aplicada al procesamiento de lenguaje natural (NLP) puede automatizar la resolución de preguntas frecuentes, reducir carga de trabajo y mejorar la experiencia del usuario.

🚀 Características

Búsqueda semántica de preguntas frecuentes (FAQ) mediante embeddings.

Respuestas automáticas en lenguaje natural.

API REST con FastAPI para integrar en aplicaciones externas.

Posibilidad de escalar a chatbots en web, Telegram o Slack.

Estructura modular y lista para expandirse (tickets, base de datos, integración con LLMs).

📂 Estructura del proyecto
soporte-tecnico-ia/
│── data/
│   └── faq.csv              # Dataset de preguntas y respuestas
│── app/
│   ├── main.py              # API con FastAPI
│   ├── nlp_engine.py        # Motor de embeddings y búsqueda
│   └── utils.py             # Funciones auxiliares
│── notebooks/
│   └── pruebas.ipynb        # Experimentos y prototipos
│── tests/
│   └── test_api.py          # Tests básicos de la API
│── requirements.txt         # Librerías necesarias
│── README.md                # Documentación del proyecto

🔧 Instalación y uso
1️⃣ Clonar el repositorio
git clone https://github.com/tuusuario/soporte-tecnico-ia.git
cd soporte-tecnico-ia

2️⃣ Crear entorno virtual e instalar dependencias
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

pip install -r requirements.txt

3️⃣ Ejecutar la API
uvicorn app.main:app --reload


La API estará disponible en:
👉 http://127.0.0.1:8000/soporte/?query=Mi%20smartwatch%20no%20carga

📊 Dataset

El dataset de ejemplo (faq.csv) contiene preguntas y respuestas simuladas, como:

“¿Cómo configuro mis auriculares inalámbricos?”

“¿Cuál es la política de devoluciones?”

“Mi smartwatch no carga, ¿qué hago?”

Se puede ampliar fácilmente con más casos reales o específicos de la empresa.

🧠 Tecnologías utilizadas

Python 3.10+

FastAPI – API REST

Sentence Transformers – Búsqueda semántica

scikit-learn / pandas – Procesamiento de datos

Uvicorn – Servidor ASGI

🔮 Mejoras futuras

Integración con un LLM (ej. GPT) para consultas no resueltas.

Sistema de tickets de soporte.

Interfaz web en React para chat en tiempo real.

Integración con Telegram / Slack / Discord.

Entrenamiento con un corpus más amplio de soporte técnico.

👨‍💻 Autor

Proyecto creado por Adrián Blasco Lozano como práctica para el desarrollo de agentes de soporte técnico con inteligencia artificial.

📌 Este repositorio forma parte de un portafolio orientado a IA, Machine Learning y Automatización de Procesos.
