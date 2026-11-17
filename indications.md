# Proyecto Final

## 🤖 Proyecto Multi-Agent ReAct

### 1. Descripción General
El objetivo de este proyecto es diseñar e implementar un sistema multi-agente autónomo basado en el paradigma de razonamiento y acción **ReAct (Reasoning and Acting)**. El sistema deberá resolver auditorías complejas o análisis de información técnica interactuando con herramientas del mundo real y bases de datos vectoriales. El grupo deberá coordinar grafos cíclicos de control, mitigar la propagación de alucinaciones y evaluar cuantitativamente la fidelidad de la recuperación de información.

### 2. Arquitectura del Sistema Obligatoria
Dado que este proyecto no actualiza los pesos de la red, la VRAM de la Tesla T4 se utilizará por completo para ejecutar la inferencia local de un modelo de tamaño medio (ej. `Mistral-7B-Instruct-v0.3` o `Llama-3-8B-Instruct` cuantizado en 4-bits) que actuará como el orquestador cognitivo del sistema:
1. **Grafo de Control de Estados (Framework de Agentes):** Uso de **LangGraph** o **CrewAI** para definir al menos 3 agentes con roles diferenciados e interdependientes (ej. *Agente Investigador*, *Agente Auditor de Hechos*, *Agente Redactor*).
2. **Bucle de Ejecución de Herramientas (Tools):** Integración de ejecución de herramientas mediante la sintaxis *Thought -> Action -> Observation*. Los agentes deben invocar dinámicamente una base de datos de conocimiento y herramientas de búsqueda web (ej. API de Tavily o DuckDuckGo Search).
3. **Mecanismo de Guardrails y Concurrencia:** Lógica de control para que el *Agente Auditor* pueda rechazar de forma condicional el reporte del *Agente Investigador*, devolviendo el estado del flujo de trabajo si detecta datos incompletos o alucinaciones.

### 3. Base de Datos y Fuentes Obligatorias
El sistema de agentes debe consultar documentos extensos en tiempo real. Implementar un motor de almacenamiento vectorial local utilizando **ChromaDB** o **FAISS** indexando el siguiente conjunto de datos disponible públicamente en Hugging Face:
* **Repositorio de Preguntas y Contextos Técnicos:** **`WikiQA`** o el corpus **`Financial-QA`**.
  * *Referencia:* Yang, Y., et al. (2015). *WikiQA: A Challenge Dataset for Open-Domain Question Answering*. Proceedings of EMNLP 2015. Disponible en Hugging Face Datasets como `'wiki_qa'`. Permite simular una base de conocimiento corporativa para auditoría de hechos.
* **Razonamiento y resolución de problemas matemáticos:** **`GSM8K` (Grade School Math Dataset)**. 
  * *Referencia:* Cobbe, K., et al. (2021). *Training Verifiers to Solve Math Word Problems*. arXiv:2110.14168. Disponible directamente en Hugging Face Datasets como `'openai/gsm8k'`.
* **Razonamiento y creación de código:** **`MBPP` (Mostly Basic Python Problems)**.
  * *Referencia:* Austin, J., et al. (2021). *Program Synthesis with Large Language Models*. arXiv:2108.07732. Disponible en Hugging Face Datasets como `'google-research-datasets/mbpp'`.
### 4. Entregables y Evaluación
* **Código:** Cuaderno `.ipynb` que dibuje el grafo de agentes y ejecute el flujo de trabajo completo.
* **Métricas de Reporte (RAGAS / LLM-as-a-Judge):** Evaluación cuantitativa sobre un set de 50 consultas complejas midiendo matemáticamente: *Faithfulness* (Fidelidad del agente al contexto), *Answer Relevance* (Relevancia de la respuesta final) y una tabla comparativa que demuestre que el sistema multi-agente supera a un sistema *Naive RAG* tradicional de un solo paso.
