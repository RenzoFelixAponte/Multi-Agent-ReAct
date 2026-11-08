# Proyecto 2: Sistema Multi-Agente ReAct (arquitectura hexagonal)

## Estructura

```
src/
├── domain/        # modelos puros (Evidence, Report, BenchmarkItem, AuditVerdict)
├── ports/         # contratos: LLMPort, WebSearchPort, KnowledgeBasePort, DatasetPort
├── application/   # núcleo: agentes ReAct, Auditor, Redactor, grafo LangGraph, Naive RAG
└── adapters/      # fakes (tests), real (Mistral 4-bit, DuckDuckGo, ChromaDB), hf_datasets
tests/             # pruebas offline con adaptadores falsos
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
