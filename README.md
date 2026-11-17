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
- **Un grafo por tarea:** `build_flow(..., task="qa" | "math" | "code")` elige las herramientas (`kb`+`web`, `calc` o `python`), el ejemplo resuelto del prompt ReAct y las reglas del Auditor y del Redactor. GSM8K y MBPP no tienen la web: no aporta y podría devolver la solución publicada del propio benchmark.
- **Guardrail:** el Auditor rechaza sin evidencia (regla determinista) o cuando el LLM no encuentra respaldo. En código no le pregunta a un LLM: **ejecuta** el código entregado con los `assert` del enunciado y, si falla, devuelve el error. El motivo y la respuesta rechazada vuelven al Investigador. Tras `max_rounds` rechazos el flujo termina sin respuesta (abstención).
- **Código entregado = código ejecutado:** si el Investigador dice que su código pasó las pruebas pero no lo copia en `Final`, se entrega el que realmente ejecutó, y el Redactor no reescribe código ya verificado.
- **Idioma:** el Investigador busca y responde en el idioma de la pregunta; la base y la web están en inglés, y las búsquedas en español recuperaban fragmentos irrelevantes.
- **Redactor sin relleno:** solo reformula la respuesta verificada, sin agregar datos ni listas de fuentes; en la prueba con la T4 inventaba explicaciones que bajaban la fidelidad.
- **Límites del 7B (medidos en la T4):** repite la misma `Action` en bucle (por eso hay una regla anti-repetición, un ejemplo en el prompt y un último paso que exige `Final`), a veces razona mal aunque `calc` le dé el resultado correcto, y como juez casi siempre da 10/10 en *Answer Relevance*. Ningún guardrail determinista arregla un error de razonamiento.
- **Base de conocimiento:** se indexan todas las frases candidatas de WikiQA, también las incorrectas, para que el Naive RAG no parta con ventaja.
- **Evaluación:** *Faithfulness* y *Answer Relevance* con las definiciones de RAGAS, implementadas como LLM-as-a-Judge (sin la librería `ragas`). GSM8K (número final) y MBPP (`test_list`) además se verifican de forma objetiva. Las abstenciones puntúan 0; `faithfulness_respondidas` promedia solo las respuestas entregadas. El juez y el Naive RAG son los mismos para los dos métodos.
- **MBPP:** ambos métodos ven los `assert` en la pregunta (como en Austin et al., 2021), pero solo el multi-agente puede ejecutarlos. Como son las mismas pruebas que se evalúan, su `exactitud` mide si sabe verificar y corregir su código, no si generaliza a pruebas ocultas.
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
