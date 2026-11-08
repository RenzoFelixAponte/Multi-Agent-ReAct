# Proyecto 2: Sistema Multi-Agente ReAct (arquitectura hexagonal)

## Estructura

```
src/proyecto2/
├── domain/        # modelos puros (Evidence, Report, AuditVerdict)
├── ports/         # contratos: LLMPort, WebSearchPort, KnowledgeBasePort
├── application/   # núcleo: agentes ReAct, Auditor, Redactor, grafo LangGraph, Naive RAG
└── adapters/      # implementaciones: fakes (tests), real (Mistral 4-bit, DuckDuckGo, ChromaDB)
tests/             # pruebas con adaptadores falsos (sin GPU ni red)
```

El núcleo (`domain` y `application`) solo importa puertos. Cambiar Mistral por otro LLM, o ChromaDB por FAISS, es reemplazar un adaptador.

## Pruebas

```bash
pip install -r requirements.txt
pytest -q
```

## Pendiente

- Validar `MistralLocalLLM` y `ChromaKB` en Colab con T4.
- Cargar WikiQA, GSM8K y MBPP desde Hugging Face (adaptador de datos).
- Notebook `notebooks/proyecto2.ipynb`: dibuja el grafo y ejecuta el flujo.
- Evaluación RAGAS / LLM-as-a-Judge sobre 50 consultas, con tabla frente a Naive RAG.
