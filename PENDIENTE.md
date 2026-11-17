# Pendiente del proyecto

Revisión de `indications.md` contra el estado actual del repo. Marcar con `[x]` al terminar cada punto.

## Falta por completo

- [ ] **Evaluación cuantitativa (sección 4).** No hay código de RAGAS ni de LLM-as-a-Judge.
  - [x] Métrica *Faithfulness* (fidelidad de la respuesta al contexto recuperado).
  - [x] Métrica *Answer Relevance* (relevancia de la respuesta final a la pregunta).
  - [x] Set de 50 consultas complejas (mezclar WikiQA, GSM8K y MBPP, o preguntas de varios pasos).
    - Hecho: `EVAL_MIX` = 20 WikiQA + 15 GSM8K + 15 MBPP (`src/application/eval_set.py`).
  - [ ] Tabla comparativa multi-agente vs Naive RAG sobre las 50 consultas. El código (`summarize`) está listo; falta la corrida real en Colab.
  - [x] Guardar resultados parciales en CSV/JSON para poder reanudar si se corta la corrida.
    - Hecho: `JsonlResultStore` escribe un registro por línea y `Evaluator.run` salta lo ya hecho y reintenta lo que falló.
- [x] **Herramientas para GSM8K y MBPP.** El `Investigator` solo entiende `kb[...]` y `web[...]` (`src/application/investigator.py:15`).
  - [x] Herramienta de cálculo para GSM8K y comparación con la referencia numérica.
  - [x] Herramienta que ejecute código Python para MBPP, en un entorno aislado y con tiempo límite.
  - [x] `map_mbpp` descarta `test_list`: conservarlo en `BenchmarkItem` para verificar el código generado.
- [ ] **LLM local en la T4.** La sección 2 exige un modelo de 7B u 8B cuantizado en 4 bits como orquestador. El notebook usa `BACKEND = "nvidia"` (API remota, modelo de 120B).
  - [ ] Validar `MistralLocalLLM` (Mistral-7B-Instruct-v0.3, 4 bits) en la T4 o en la laptop (RTX 4050, 6 GB: cabe justo).
  - [x] Dejar `BACKEND = "mistral"` como valor por defecto en el notebook entregable.
  - [ ] Si el modelo es restringido en Hugging Face, aceptar los términos y usar `HF_TOKEN`.

## Está roto o incompleto

- [x] **El notebook falla en la celda 10.** Usa `HFDatasetAdapter`, `wikiqa_documents`, `build_flow`, `naive_rag` y `pandas` sin importarlos.
- [ ] **El notebook no está ejecutado.** Todas las celdas tienen 0 outputs; el entregable debe dibujar el grafo y ejecutar el flujo.
- [x] **Texto desactualizado.** La celda 0 dice que "el push está pendiente", pero `main` ya está sincronizada con `origin/main`.
- [x] **`requirements.txt` incompleto.** Faltan `datasets`, `pandas`, `chromadb`, `ddgs`, `transformers`, `accelerate`, `bitsandbytes`, `torch` y la librería de métricas. Casi todo está comentado.
  - Las métricas son una implementación propia (LLM-as-a-Judge al estilo RAGAS), así que no hay librería extra.
- [x] **README.** Actualizar la sección "Pendiente" cuando se avance.

## Puntos a revisar

- [x] **Concurrencia.** El título 2.3 dice "Guardrails y Concurrencia" y no hay nada concurrente en el código (por ejemplo, consultar `kb` y `web` en paralelo con fan-out de LangGraph). Implementarlo o justificar su ausencia en el informe.
  - Hecho: `build_flow(..., prefetch=True)` consulta `kb` y `web` en paralelo (fan-out de LangGraph, `tests/test_concurrency.py` lo comprueba con una barrera) y el Investigador arranca con esa evidencia.
- [x] **Sesgo en la comparación con Naive RAG.** `map_wikiqa` solo conserva frases con `label=1`, así que la KB contiene únicamente respuestas correctas y el Naive RAG parte con ventaja. Indexar también las frases `label=0` como distractores, o incluir preguntas que necesiten búsqueda web.
  - Hecho: `group_wikiqa` agrupa por pregunta y la KB indexa todas las frases candidatas (WikiQA validation: 126 preguntas con respuesta, 1119 frases únicas).
- [x] **Juez de la evaluación.** Decidir qué LLM actúa como juez (el mismo Mistral local u otro). Un 7B como juez es ruidoso; documentar la elección.
  - Decisión: por defecto el mismo Mistral (`JUDGE_BACKEND = "same"`), con opción de juez remoto; documentado en el README y en el notebook, y los formatos inválidos se cuentan en `juez_fallido`.
- [ ] **Corrida de 50 consultas.** Puede tomar de 1 a 3 horas con Mistral 4 bits (estimación, sin medir). Medir el tiempo con 3 a 5 consultas antes de lanzar todo.

## Dónde ejecutar

- Desarrollo y pruebas: laptop con `ScriptedLLM` o `NvidiaLLM` (no necesita GPU).
- Validación del modelo local: laptop (RTX 4050, 6 GB de VRAM) o Colab (T4, 16 GB). Falta `torch` en la laptop; usar un venv.
- Corrida final: Colab con T4. Opción: [Google Colab CLI](https://github.com/googlecolab/google-colab-cli) oficial (Linux/macOS).
  - [x] Instalar con `uv tool install google-colab-cli` y hacer el login OAuth (lo hace el usuario).
  - [ ] Comprobar si la cuenta tiene T4: `colab new --gpu T4 -s proyecto` (un `--gpu` mal escrito cae en A100 sin avisar).
  - [x] Cambiar el notebook para usar el código subido en vez de clonar el repo de GitHub. Usa `/content/Multi-Agent-ReAct` si existe; `scripts/colab_run.sh` lo sube con `colab upload`.
  - [ ] Ejecutar con `colab exec -s proyecto -f notebooks/proyecto2.ipynb`, bajar `_output.ipynb` y el CSV, y cerrar con `colab stop -s proyecto`.
  - [ ] Verificar con `colab help exec` si existe modo en segundo plano y cómo pasar `HF_TOKEN` (`--env`); el README oficial no lo documenta.

## Ya cumple

- Grafo cíclico en LangGraph con 3 agentes: Investigador, Auditor y Redactor.
- Bucle Thought → Action → Observation.
- Rechazo condicional del Auditor con reintentos limitados.
- ChromaDB local, búsqueda web con DuckDuckGo y baseline `naive_rag`.
- Loaders de WikiQA, GSM8K y MBPP desde Hugging Face.
