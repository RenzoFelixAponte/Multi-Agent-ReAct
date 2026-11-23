# Análisis de errores de la corrida en la T4

Revisión, consulta por consulta, de los 100 registros de [`results/evaluacion.jsonl`](../results/evaluacion.jsonl): 50 consultas con los dos métodos, Mistral-7B-Instruct-v0.3 de 4 bits y el mismo modelo como juez. Las cifras se reproducen con:

```bash
python scripts/analisis_errores.py
```

## Resumen

1. **El guardrail funciona cuando es determinista.** En MBPP el Auditor ejecuta el código con sus pruebas, y el multi-agente no entregó ninguna respuesta incorrecta, contra 9 del Naive RAG. En GSM8K el Auditor es un LLM y aprobó 7 respuestas incorrectas.
2. **Las abstenciones evitan errores.** En 7 de las 8 abstenciones del multi-agente, el Naive RAG entregó una respuesta incorrecta.
3. **El juez de 7B casi no distingue una respuesta correcta de una incorrecta.** Por eso *Faithfulness* y *Answer Relevance* dicen poco sobre la calidad en GSM8K y MBPP, y la verificación objetiva es la evidencia más fuerte de la comparación.
4. **La ventaja en exactitud es consistente pero no significativa con 30 consultas.** Hay 8 aciertos discordantes a favor del multi-agente y 2 a favor del Naive RAG (McNemar exacto, p = 0.109).

## 1. Resultado por consulta en GSM8K y MBPP

| multi-agente | Naive RAG | GSM8K | MBPP | Total |
|---|---|---|---|---|
| correcta | correcta | 1 | 6 | 7 |
| correcta | incorrecta | 4 | 4 | **8** |
| incorrecta | correcta | 1 | 0 | **1** |
| incorrecta | incorrecta | 6 | 0 | 6 |
| abstención | correcta | 1 | 0 | 1 |
| abstención | incorrecta | 2 | 5 | 7 |

- El multi-agente acierta 8 consultas que el Naive RAG falla. El Naive RAG acierta 2 que el multi-agente no acierta: una por error y otra por abstención.
- Con la prueba exacta de McNemar sobre esos 10 pares discordantes, p = 0.109. La dirección es clara, pero 30 consultas no alcanzan para descartar el azar con un nivel de 0.05.
- En respuestas incorrectas entregadas, el multi-agente tiene 7 de 30 (23 %) y el Naive RAG 21 de 30 (70 %). Esta es la diferencia más grande entre los dos métodos.

## 2. El Auditor: qué pasó con cada rechazo

| Dataset | Con al menos un rechazo | Corregidas tras el rechazo | Abstenciones | Aprobadas pero incorrectas |
|---|---|---|---|---|
| WikiQA | 1 de 20 | 1 (sin verificación objetiva) | 0 | — |
| GSM8K | 3 de 15 | 0 | 3 | **7** |
| MBPP | 7 de 15 | 2 (ambas correctas) | 5 | **0** |

**MBPP: auditoría por ejecución.** El Auditor no le pregunta al LLM: ejecuta el código entregado con los `assert` del enunciado. Por eso nunca aprobó código erróneo. En dos casos el error de ejecución devuelto al Investigador le permitió corregir su código:

- `find_first_duplicate` (MBPP 22) se corrigió en la segunda ronda.
- `find_Product` (MBPP 25) se corrigió en la tercera ronda. El Naive RAG falló las dos.

**GSM8K: auditoría con el LLM.** El Auditor solo puede leer el razonamiento, y el mismo 7B que se equivocó suele aprobar su propio error. De las 7 respuestas aprobadas que el verificador marcó como incorrectas, 6 son errores de razonamiento, no de aritmética, porque `calc` calcula bien lo que se le pide. La séptima es un falso negativo del verificador:

| # | Referencia | Respuesta | Error |
|---|---|---|---|
| 2 | 70 000 | 30 000 | Interpreta "el valor aumentó un 150 %" como multiplicar por 1.5 en vez de por 2.5 |
| 6 | 260 | 240 | Plantea bien la suma (160 + 80 + 20) pero da 240 |
| 7 | 160 | 120 | Interpreta mal "reiniciar la descarga desde el principio" |
| 10 | 366 | 204 | Calcula bien los tres meses (60, 180 y 126) pero los suma mal |
| 13 | 18 | 6 | No plantea la ecuación hacia atrás |
| 14 | 60 | 50 | Deja un `calc[...]` sin ejecutar dentro de la respuesta final |
| 5 | 64 | 64 | **Falso negativo del verificador** (ver la sección 4) |

Los casos 6 y 10 muestran un límite conocido del 7B: escribe bien la operación y luego da otro resultado sin pasarla por `calc`. Ningún guardrail con el mismo modelo lo detecta de forma confiable.

## 3. El juez frente a la verificación objetiva

Sobre las 52 respuestas entregadas en GSM8K y MBPP por los dos métodos:

| Respuestas | n | *Answer Relevance* | *Faithfulness* |
|---|---|---|---|
| correctas | 24 | 1.000 | 0.735 |
| incorrectas | 28 | 0.927 | 0.694 |

El juez le da casi el mismo puntaje a una respuesta correcta que a una incorrecta. Es esperable en *Answer Relevance*, que por definición no juzga si la respuesta es verdadera. En *Faithfulness* también falla:

- El código correcto de MBPP 25, que pasó todas las pruebas al ejecutarse, recibió fidelidad 0.0.
- Dos códigos incorrectos del Naive RAG recibieron 1.0.

Esto explica por qué las métricas del juez no separan bien a los dos métodos. Por eso la tabla agrega `exactitud_respondidas` e `incorrectas_entregadas`, que salen de la verificación objetiva y no del juez.

## 4. Falso negativo del verificador de GSM8K

`extract_final_number` toma el número que sigue a "Respuesta final:" y, si esa línea no existe, el último número del texto. En GSM8K 5 el multi-agente respondió:

> Kylar necesita pagar $64 por los 16 vasos. [...] Esto se expresa matemáticamente como: 8\*5 + 8\*3.

La respuesta es correcta, pero el Redactor no escribió la línea "Respuesta final:" que le pide su regla, y el verificador tomó el 3. Se revisó a mano el número extraído de las 30 respuestas de GSM8K (15 por método) contra la referencia, y es el único caso.

**Los números reportados no se corrigieron:** se dejó la salida del verificador automático tal cual. Con la corrección manual, la exactitud del multi-agente en GSM8K sería 0.40 en vez de 0.33, y la global 0.53 en vez de 0.50.

## 5. WikiQA: fidelidad baja en respuestas correctas

En WikiQA no hay verificación objetiva. La fidelidad promedio es casi igual entre los métodos (0.605 contra 0.649), pero el multi-agente tiene varias respuestas cortas y correctas con fidelidad muy baja:

| Pregunta | Respuesta del multi-agente | *Faithfulness* |
|---|---|---|
| what bird family is the owl | La familia del búho es Strigidae. | 0.11 |
| what county is Holly Ridge nc in? | Holly Ridge, North Carolina is located in Onslow County. | 0.12 |
| what kind of literature did john steinbeck writing | John Steinbeck writes realism literature. | 0.14 |

| Método | Respuestas cortas (< 80 caracteres) | Respuestas largas | Fragmentos de contexto |
|---|---|---|---|
| multi-agente | 9 con 0.483 | 11 con 0.705 | 7.05 en promedio |
| Naive RAG | 5 con 0.670 | 15 con 0.641 | 3.0 |

La hipótesis más probable es que el juez rinde peor cuando el contexto es largo. El multi-agente le pasa entre 6 y 10 fragmentos (los de `kb`, los de `web` y los de sus búsquedas), y el juez tiene que encontrar la frase que respalda una afirmación corta. No se probó: habría que volver a juzgar con contextos de igual tamaño o con un juez más fuerte, y los registros no guardan el texto de la evidencia.

No es un efecto del idioma. El multi-agente respondió en español a 7 de las 20 preguntas en inglés, pero esas respuestas tienen mayor fidelidad promedio (0.77) que las respuestas en inglés (0.52).

## 6. Otros hallazgos

- **Idioma de la respuesta.** La regla pide responder en el idioma de la pregunta, pero el Redactor respondió en español a 7 preguntas en inglés, probablemente porque su prompt está en español.
- **Abstenciones en MBPP.** Las 5 abstenciones agotaron las 3 rondas sin una respuesta aprobada. Los registros no guardan el motivo de cada rechazo, así que no se puede saber si el problema fue el algoritmo o el formato.
- **El juez falló en 4 respuestas** de MBPP del Naive RAG (columna `juez_fallido`), siempre en *Answer Relevance*. Esas respuestas quedan fuera del promedio de esa métrica.

## Qué haría falta para confirmar estas conclusiones

1. **Más consultas.** Con unas 100 de GSM8K y MBPP, una diferencia como la observada probablemente sería significativa.
2. **Guardar la evidencia y el motivo de cada rechazo** en `EvalRecord`, para volver a juzgar sin repetir la generación y para estudiar las abstenciones.
3. **Un juez independiente y más fuerte** (`JUDGE_BACKEND = "nvidia"`), para separar la calidad del sistema del ruido del juez.
4. **Una ablación del Auditor** (`max_rounds = 1`) para medir cuánto aporta cada agente.
