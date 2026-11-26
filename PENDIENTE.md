# Pendiente del proyecto

Revisión de `indications.md` contra el estado actual del repo. Marcar con `[x]` al terminar cada punto.

## Falta por completo

- [x] **Evaluación cuantitativa (sección 4).** No hay código de RAGAS ni de LLM-as-a-Judge.
  - [x] Métrica *Faithfulness* (fidelidad de la respuesta al contexto recuperado).
  - [x] Métrica *Answer Relevance* (relevancia de la respuesta final a la pregunta).
  - [x] Set de 50 consultas complejas (mezclar WikiQA, GSM8K y MBPP, o preguntas de varios pasos).
    - Hecho: `EVAL_MIX` = 20 WikiQA + 15 GSM8K + 15 MBPP (`src/application/eval_set.py`).
  - [x] Tabla comparativa multi-agente vs Naive RAG sobre las 50 consultas.
    - Hecho: `results/tabla_comparativa.csv` y la sección "Resultados" del README. El multi-agente gana en exactitud (0.50 contra 0.30) y en fidelidad cuando responde (0.691 contra 0.663), pero pierde en las métricas promediadas sobre las 50 porque sus 8 abstenciones puntúan 0.
  - [x] Guardar resultados parciales en CSV/JSON para poder reanudar si se corta la corrida.
    - Hecho: `JsonlResultStore` escribe un registro por línea y `Evaluator.run` salta lo ya hecho y reintenta lo que falló.
- [x] **Herramientas para GSM8K y MBPP.** El `Investigator` solo entiende `kb[...]` y `web[...]` (`src/application/investigator.py:15`).
  - [x] Herramienta de cálculo para GSM8K y comparación con la referencia numérica.
  - [x] Herramienta que ejecute código Python para MBPP, en un entorno aislado y con tiempo límite.
  - [x] `map_mbpp` descarta `test_list`: conservarlo en `BenchmarkItem` para verificar el código generado.
- [x] **LLM local en la T4.** La sección 2 exige un modelo de 7B u 8B cuantizado en 4 bits como orquestador. El notebook usa `BACKEND = "nvidia"` (API remota, modelo de 120B).
  - [x] Validar `MistralLocalLLM` (Mistral-7B-Instruct-v0.3, 4 bits) en la T4 o en la laptop (RTX 4050, 6 GB: cabe justo).
    - Hecho en Colab: carga en una `Tesla T4` y corre el flujo completo de las tres tareas. Con la cuenta gratuita (0 unidades de cómputo) la T4 estuvo disponible.
  - [x] Dejar `BACKEND = "mistral"` como valor por defecto en el notebook entregable.
  - [x] Si el modelo es restringido en Hugging Face, aceptar los términos y usar `HF_TOKEN`.

## Está roto o incompleto

- [x] **El notebook falla en la celda 10.** Usa `HFDatasetAdapter`, `wikiqa_documents`, `build_flow`, `naive_rag` y `pandas` sin importarlos.
- [x] **El notebook no está ejecutado.** Todas las celdas tienen 0 outputs; el entregable debe dibujar el grafo y ejecutar el flujo.
  - Hecho: `notebooks/proyecto2_output.ipynb` (ejecutado en la T4, con el grafo dibujado y la tabla).
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
- [x] **Corrida de 50 consultas.** Puede tomar de 1 a 3 horas con Mistral 4 bits (estimación, sin medir). Medir el tiempo con 3 a 5 consultas antes de lanzar todo.
  - Medido en la T4: 12 registros en 501 a 665 s (42 a 55 s por registro, 24 s Naive RAG y ~60 s multi-agente), o sea 70 a 90 min para los 100 registros.
  - Hecho: la corrida completa tomó unos 96 min (89 s por consulta el multi-agente y 26 s el Naive RAG), sin errores.

## Dónde ejecutar

- Desarrollo y pruebas: laptop con `ScriptedLLM` o `NvidiaLLM` (no necesita GPU).
- Validación del modelo local: laptop (RTX 4050, 6 GB de VRAM) o Colab (T4, 16 GB). Falta `torch` en la laptop; usar un venv.
- Corrida final: Colab con T4. Opción: [Google Colab CLI](https://github.com/googlecolab/google-colab-cli) oficial (Linux/macOS).
  - [x] Instalar con `uv tool install google-colab-cli` y hacer el login OAuth (lo hace el usuario).
  - [x] Comprobar si la cuenta tiene T4: `colab new --gpu T4 -s proyecto` (un `--gpu` mal escrito cae en A100 sin avisar).
  - [x] Cambiar el notebook para usar el código subido en vez de clonar el repo de GitHub. Usa `/content/Multi-Agent-ReAct` si existe; `scripts/colab_run.sh` lo sube con `colab upload`.
  - [x] Ejecutar con `colab exec -s proyecto -f notebooks/proyecto2.ipynb`, bajar `_output.ipynb` y el CSV, y cerrar con `colab stop -s proyecto`.
  - [x] Verificar con `colab help exec` si existe modo en segundo plano y cómo pasar `HF_TOKEN` (`--env`); el README oficial no lo documenta.
    - `--env KEY=VALUE` funciona y `--timeout 0` quita el límite de silencio. No hay modo en segundo plano: se lanza el comando local en segundo plano. `echo '!cmd' | colab exec` se colgó una vez; usar `-f archivo.py`.

## Mejoras de la entrega

- [x] README reorganizado: diagrama del grafo, tablas y secciones navegables.
- [x] Métricas de alucinación (`exactitud_respondidas`, `incorrectas_entregadas`) sin volver a correr: `scripts/resumir_resultados.py`.
- [x] Análisis de errores consulta por consulta: `docs/analisis_errores.md` y `scripts/analisis_errores.py`.
- [x] Resultados y conclusiones del README actualizados.
- [x] Informe final en LaTeX: `informe/` (compila en Overleaf y con tectonic).
  - [x] Poner los nombres de los integrantes y el curso en `informe/main.tex`.
- [ ] Opcional: ablación del Auditor (`max_rounds=1`) y de `prefetch` (necesita GPU: Colab o la PC con la RTX 5070).
- [ ] Opcional: guardar la evidencia en `EvalRecord` y volver a juzgar con un juez independiente (`JUDGE_BACKEND = "nvidia"`). Pasa a la corrida 2.
- [ ] Opcional: `test_runner_limits_memory` falla en macOS (allí no se aplica `RLIMIT_AS`); en Linux y Colab pasa.

## Corrida 2: Qwen2.5-7B y juez independiente

Con Mistral-7B el multi-agente pierde en *Faithfulness* (0.580 contra 0.663) y *Answer Relevance* (0.792 contra 0.911) promediadas sobre las 50. En las 42 consultas que ambos responden gana en las dos (0.691 contra 0.654 y 0.938 contra 0.910): la diferencia la hacen sus 8 abstenciones, que puntúan 0. Además, el juez de 7B da relevancia 1.0 y fidelidad 0.86 a las respuestas incorrectas del multi-agente.

- [x] Idioma explícito en el Investigador, el Redactor y el Naive RAG: 9 de 20 respuestas de WikiQA salieron en español.
- [x] Guardar la traza ReAct (`Report.trace`) y los motivos de rechazo (`FlowState.rejections`).
- [x] La ejecución de las pruebas por el Auditor cuenta como evidencia (MBPP 25: código correcto con fidelidad 0).
- [x] Guardar evidencia, traza y rechazos en `EvalRecord`; el juez que falla no debe perder la generación.
- [x] Reintentos con espera en `NvidiaLLM` (429 y 5xx) y `scripts/rejuzgar.py` para lo que quede sin juzgar.
- [x] Orquestador `Qwen/Qwen2.5-7B-Instruct` en 4 bits, juez remoto de NVIDIA y `max_rounds = 5` en MBPP.
- [x] Resultados de cada modelo en su carpeta (`results/<modelo>/`), para no saltarse consultas de otra corrida.
- [x] Prueba de tiempo (`FULL_RUN=0`) y corrida completa en la T4.
  - Hecho el 26/11/2026: 100 registros en unos 100 min, sin errores. Resultados en `results/qwen2.5-7b-instruct/`.
  - El multi-agente gana en *Faithfulness* (0.804 contra 0.678) y en respuestas incorrectas entregadas (0.10 contra 0.17), y gana en WikiQA y MBPP. Pierde en *Answer Relevance* (0.860 contra 0.960) y en exactitud (0.733 contra 0.833) por GSM8K: 9/15 contra 14/15.
  - El registro de MBPP 12 (Naive RAG) quedó sin puntaje del juez y se volvió a juzgar con `scripts/rejuzgar.py`: `tabla_comparativa.csv` lo incluye y la tabla del notebook ejecutado no.
- [ ] GSM8K: el Auditor (LLM) rechaza respuestas correctas. En las consultas 1, 13 y 14 se abstuvo donde el Naive RAG acertó; en la 1 el propio Auditor calcula 3, que es la respuesta correcta, y la rechaza. Propuesta: que el Auditor resuelva el problema por su cuenta y compare el número, y que el Investigador razone paso a paso antes de `calc`. Validar con otras preguntas de GSM8K, no con las 15 del set, para no sobreajustar.
- [ ] Actualizar README, informe y análisis de errores con la corrida 2.
- [x] Presentación en LaTeX con la plantilla de Computer Vision (`presentacion/`, 10 diapositivas) y su guion (`presentacion/guion.md`), combinada con la versión de 18 diapositivas del grupo (commit `dea820f`). Falta poner el resultado de GSM8K.

## Ya cumple

- Grafo cíclico en LangGraph con 3 agentes: Investigador, Auditor y Redactor.
- Bucle Thought → Action → Observation.
- Rechazo condicional del Auditor con reintentos limitados.
- ChromaDB local, búsqueda web con DuckDuckGo y baseline `naive_rag`.
- Loaders de WikiQA, GSM8K y MBPP desde Hugging Face.
