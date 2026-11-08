# Proyecto 2: Sistema Multi-Agente ReAct (arquitectura hexagonal)

## Estructura

```
src/
├── domain/        # entidades puras, una por archivo: Evidence, Report, BenchmarkItem, AuditVerdict
├── ports/         # contratos (Protocol), uno por archivo: LLMPort, WebSearchPort, KnowledgeBasePort, DatasetPort
├── application/   # núcleo: Investigator, Auditor, Redactor, FlowState, build_flow, naive_rag
│                  #   prompts.py y formatting.py tienen las plantillas y el formato, sin lógica
└── adapters/      # implementaciones, una clase por archivo:
                   #   reales: MistralLocalLLM, NvidiaLLM, DuckDuckGoSearch, ChromaKB, HFDatasetAdapter
                   #   falsas (pruebas): ScriptedLLM, StaticSearch, InMemoryKB
                   #   utilidades: dotenv_loader, hf_mappers
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
