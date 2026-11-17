# Proyecto 2: Sistema Multi-Agente ReAct (arquitectura hexagonal)

Tres agentes en un grafo cíclico de LangGraph: **Investigador** (bucle ReAct *Thought → Action → Observation*), **Auditor de Hechos** (guardrail que rechaza y devuelve el flujo) y **Redactor**. El orquestador es Mistral-7B-Instruct-v0.3 cuantizado a 4 bits en la T4 de Colab.

```
        ┌─ kb ──┐
inicio ─┤       ├─> Investigador ─> Auditor ─┬─ aprobado ──> Redactor ─> fin
        └─ web ─┘        ^                   │
      (en paralelo)      └─── rechazo + motivo (hasta max_rounds)
```

Herramientas del Investigador: `kb[...]` (ChromaDB sobre WikiQA), `web[...]` (DuckDuckGo), `calc[...]` (GSM8K) y `python[...]` (MBPP, en un proceso aislado con tiempo y memoria limitados).

## Estructura

```
src/
├── domain/        # entidades puras, una por archivo: Evidence, Report, BenchmarkItem, AuditVerdict,
│                  #   ExecutionResult, EvalRecord
├── ports/         # contratos (Protocol): LLMPort, WebSearchPort, KnowledgeBasePort, DatasetPort,
│                  #   CodeRunnerPort, ResultStorePort
├── application/   # núcleo: Investigator, Auditor, Redactor, build_flow, naive_rag
│                  #   herramientas: calculator, answer_checks
│                  #   evaluación: Judge, Evaluator, eval_set, eval_summary
│                  #   prompts.py y formatting.py tienen las plantillas y el formato, sin lógica
└── adapters/      # implementaciones, una clase por archivo:
                   #   reales: MistralLocalLLM, NvidiaLLM, DuckDuckGoSearch, ChromaKB, HFDatasetAdapter,
                   #           SubprocessCodeRunner, JsonlResultStore
                   #   falsas (pruebas): ScriptedLLM, StaticSearch, InMemoryKB
                   #   utilidades: dotenv_loader, hf_mappers
notebooks/         # proyecto2.ipynb: dibuja el grafo, ejecuta el flujo y la evaluación
tests/             # pruebas offline con adaptadores falsos
```

El núcleo (`domain` y `application`) solo importa puertos. Cambiar Mistral por otro LLM, o ChromaDB por FAISS, es reemplazar un adaptador.

## Decisiones de diseño

- **Concurrencia:** `build_flow(..., prefetch=True)` consulta `kb` y `web` a la vez (fan-out de LangGraph, con un reductor que une la evidencia) antes del bucle ReAct. Una rama caída no tumba a la otra.
- **Guardrail:** el Auditor rechaza sin evidencia o con código que no se ejecutó tal cual con `python[...]` (reglas deterministas), o cuando el LLM no encuentra respaldo; el motivo vuelve al Investigador. Tras `max_rounds` rechazos el flujo termina sin respuesta (abstención). El Redactor no reescribe el código ya verificado.
- **Límites del 7B (medidos en la T4):** repite la misma `Action` en bucle (por eso hay una regla anti-repetición, un ejemplo en el prompt y un último paso que exige `Final`), a veces razona mal aunque `calc` le dé el resultado correcto, y como juez casi siempre da 10/10 en *Answer Relevance*. Ningún guardrail determinista arregla un error de razonamiento.
- **Base de conocimiento:** se indexan todas las frases candidatas de WikiQA, también las incorrectas, para que el Naive RAG no parta con ventaja.
- **Evaluación:** *Faithfulness* y *Answer Relevance* con las definiciones de RAGAS, implementadas como LLM-as-a-Judge (sin la librería `ragas`). GSM8K (número final) y MBPP (`test_list`) además se verifican de forma objetiva. Las abstenciones puntúan 0.
- **Juez:** por defecto es el mismo Mistral de 7B que orquesta (`JUDGE_BACKEND = "same"` en el notebook); un 7B como juez es ruidoso y puede favorecer sus propias respuestas. Se puede usar un juez remoto con `JUDGE_BACKEND = "nvidia"`. Los formatos inválidos del juez se cuentan en `juez_fallido`.
- **Código ejecutado:** `SubprocessCodeRunner` limita tiempo, CPU y memoria, pero no bloquea la red ni los archivos; es aceptable porque la VM de Colab es desechable.

## Pruebas locales

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python langgraph datasets pytest
.venv/bin/python -m pytest -q
```

Las pruebas no necesitan GPU ni red. Para correr el notebook sin GPU, `BACKEND = "nvidia"` usa la API remota (`NVIDIA_API_KEY` en `.env`, ver `.env.example`).

## Ejecutar en Colab (T4)

Desde el navegador: abrir `notebooks/proyecto2.ipynb` en Colab con GPU T4, poner `HF_TOKEN` en *Secrets* (Mistral es un modelo restringido: hay que aceptar sus términos en Hugging Face) y ejecutar todo.

Desde la terminal con la [Colab CLI](https://github.com/googlecolab/google-colab-cli) (el primer comando abre el login OAuth en el navegador):

```bash
HF_TOKEN=hf_... scripts/colab_run.sh              # corrida completa
FULL_RUN=0 HF_TOKEN=hf_... scripts/colab_run.sh   # solo la prueba de tiempo
```

El script crea una sesión con T4, sube `src/`, ejecuta el notebook (la salida queda en `notebooks/proyecto2_output.ipynb`), baja los resultados a `results/` y cierra la sesión.

La corrida guarda cada resultado en `results/evaluacion.jsonl`; si se corta, volver a ejecutarla reanuda donde quedó.

## Pendiente

Ver `PENDIENTE.md`.
