# Guion de la presentación

10 diapositivas, unos 10 minutos. El reparto es una propuesta: cada integrante se queda con un bloque seguido, para no pasarse la palabra a cada rato.

| Bloque | Diapositivas | Quién | Tiempo |
|---|---|---|---|
| Contexto y sistema | 1 a 4 | Renzo | 3 min 40 s |
| Herramientas y evaluación | 5 a 7 | Jose | 3 min 10 s |
| Hallazgos y cierre | 8 a 10 | Diego | 3 min 10 s |

Todos los números de las diapositivas salen de `results/` (ver el encabezado de `main.tex`). Al final hay respuestas preparadas para las preguntas más probables.

---

## 1. Portada (Renzo, 20 s)

> Buenas. Somos Renzo, Jose y Diego. Les vamos a presentar un sistema multi-agente ReAct: tres agentes que no solo responden, sino que revisan la respuesta antes de entregarla. El fondo de la portada son trazas reales del sistema razonando.

## 2. El problema y lo que pedía el enunciado (Renzo, 1 min)

> El punto de partida: un modelo de 7B, como los que caben en una T4, responde con la misma seguridad aunque esté equivocado. En nuestra primera corrida, el Naive RAG entregó 21 respuestas incorrectas de las 30 que podíamos verificar.
>
> A la derecha está lo que pedía el enunciado y cómo lo resolvimos: tres agentes en un grafo cíclico de LangGraph, el ciclo Thought, Action, Observation con herramientas, ChromaDB y búsqueda web, un Auditor que rechaza, consultas en paralelo, un modelo de 7B en 4 bits en la T4 y 50 consultas contra un Naive RAG. Todo está cubierto.

## 3. Un grafo cíclico con tres agentes (Renzo, 1 min)

> La pregunta entra y, en WikiQA, se consultan la base vectorial y la web al mismo tiempo: esa es la parte concurrente. El **Investigador** arranca con esa evidencia y sigue su bucle ReAct, usando herramientas. Su respuesta pasa al **Auditor de Hechos**. Si la aprueba, el **Redactor** la deja en limpio. Si la rechaza, el motivo vuelve al Investigador para que lo intente otra vez: ese es el ciclo.
>
> Si el Auditor rechaza tres veces, el sistema se abstiene en vez de entregar algo sin respaldo. En MBPP permitimos cinco intentos, porque ahí reintentar no puede colar un error.

## 4. El bucle ReAct y un rechazo real (Renzo, 1 min 20 s)

> Este es un caso real de la corrida, el problema 11 de MBPP: una función que quite la primera y la última aparición de un carácter, con tres pruebas.
>
> En la primera ronda, el Investigador entrega una función. El Auditor no le pregunta a un LLM si el código "parece" correcto: lo **ejecuta** con las pruebas del enunciado, y la segunda prueba falla.
>
> Ese error vuelve al Investigador. En la segunda ronda, su Thought reconoce que el problema era usar `replace`, cambia a `find` y `rfind`, ejecuta el código con las pruebas y la Observation dice "ok". La Observation la escribe siempre el sistema, nunca el modelo, porque los 7B tienden a inventarse el resultado de la herramienta. Ahora el Auditor aprueba.

## 5. Un grafo por tarea, ajustado en la T4 (Jose, 1 min)

> Cada tarea tiene su grafo. En las preguntas de WikiQA, el Investigador busca en la base y en la web, y el Auditor contrasta la respuesta con la evidencia. En GSM8K usa una calculadora que no usa `eval`: recorre el árbol sintáctico. En MBPP, Python en un proceso aparte con límites. A matemáticas y código les quitamos la web a propósito: podría devolver la solución publicada del benchmark.
>
> Abajo está lo más práctico: problemas que aparecieron al correr en la T4 y que no previmos. El modelo repetía la misma acción en bucle; copiaba mal su propio código al responder, así que ahora entregamos el que de verdad se ejecutó; el Redactor inventaba explicaciones; y respondía en español a preguntas en inglés. Cada arreglo salió de leer las trazas.

## 6. Cómo evaluamos (Jose, 1 min)

> Las 50 consultas son las mismas para los dos métodos. La línea base es un **Naive RAG**: una recuperación y una sola llamada al mismo modelo, sin herramientas ni Auditor.
>
> Las dos métricas del enunciado, al estilo RAGAS, las calcula un juez: **Faithfulness** es la fracción de afirmaciones de la respuesta que se pueden inferir del contexto, y **Answer Relevance**, qué tan directo responde la pregunta. Como en matemáticas y código sí se puede comprobar la respuesta, agregamos la exactitud y una métrica que nos parece clave: cuántas respuestas **incorrectas** llegan al usuario. Una abstención vale cero para el juez, pero no cuenta como error.

## 7. Resultados con Qwen2.5-7B y el juez de 120B (Jose, 1 min 10 s)

> Con el juez independiente, el multi-agente gana en **fidelidad**, 0.80 contra 0.68, y entrega menos respuestas incorrectas: 10 % contra 17 %. Pierde en relevancia, por sus abstenciones, que valen cero, y en exactitud.
>
> A la derecha está de dónde sale cada cosa. En WikiQA gana en fidelidad. En MBPP acierta 13 contra 11 y no entregó ningún código incorrecto; el Naive RAG entregó 4. Y en GSM8K pierde: 9 aciertos contra 14. Eso lo explica Diego.

## 8. El juez cambia la conclusión (Diego, 1 min)

> Nuestra primera corrida fue con Mistral-7B, y el juez era el mismo Mistral. Ahí el multi-agente perdía en las dos métricas del enunciado. Pero ese juez casi no medía: daba fidelidad 0.74 a las respuestas correctas y 0.69 a las incorrectas. El juez de 120B sí las separa: 0.85 contra 0.54.
>
> El enunciado exige que el orquestador sea local, no el juez, así que lo hicimos independiente. Una advertencia honesta: en la segunda corrida cambiamos a la vez el modelo y el juez, así que no podemos atribuir la mejora a uno solo.

## 9. Un guardrail vale lo que su verificación (Diego, 1 min 10 s)

> Esta es la lección principal. Comparen los dos Auditores. En MBPP, el Auditor **ejecuta** el código: no dejó pasar ni un código incorrecto con ninguno de los dos modelos.
>
> En GSM8K, el Auditor **opinaba** en texto libre. Miren este caso real: la respuesta correcta era 3. El Auditor escribe que el cálculo debe ser 2 más 2 entre 2, igual a 3, "lo cual es correcto"... y la rechaza. Tres veces, así que el sistema se abstuvo, y el Naive RAG respondió 3.
>
> Lo cambiamos: ahora el Auditor resuelve el problema por su cuenta, sin ver la respuesta, y solo rechaza si los números no coinciden.

## 10. Lo que aprendimos (Diego, 1 min)

> Cuatro lecciones. Uno: verificar ejecutando, no opinando. Dos: el juez cambia la conclusión, así que hay que medir con uno independiente. Tres: abstenerse baja las métricas promediadas, pero evita entregar errores. Y cuatro: más andamiaje no siempre ayuda; Qwen ya resolvía GSM8K solo, y un mal Auditor le restaba.
>
> Las limitaciones están a la derecha: con 50 consultas, ninguna diferencia es estadísticamente significativa, y el multi-agente es unas cuatro veces más lento. Todo el código, el notebook y los resultados están en el repositorio. Gracias, ¿preguntas?

---

## Preguntas probables

**¿Por qué el multi-agente pierde en Answer Relevance si es "mejor"?**
Por las abstenciones, que valen 0. En las 44 consultas que responden los dos, el multi-agente gana: 0.977 contra 0.973. No escondimos las abstenciones: la tabla las cuenta y reporta aparte la fracción respondida.

**¿No es trampa cambiar el juez después de ver los resultados?**
Lo cambiamos porque el juez de 7B no distinguía respuestas correctas de incorrectas, y eso lo comprobamos con la verificación objetiva de GSM8K y MBPP. El juez nuevo es el mismo para los dos métodos, y los resultados con Mistral siguen en el repositorio.

**¿El cambio del Auditor de GSM8K no está ajustado a sus 15 preguntas?**
Lo validamos antes en 20 problemas del split *train* de GSM8K, que no están en la evaluación (que usa el split *test*), comparando el código anterior con el nuevo. Están en `results/validacion_gsm8k/`.

**¿Por qué el Naive RAG resuelve tan bien GSM8K?**
Porque Qwen2.5-7B razonando paso a paso ya es muy bueno en GSM8K. Con Mistral era al revés: el Naive RAG acertaba 0.30 y el multi-agente 0.50.

**¿Es significativa la diferencia?**
No. Con 30 consultas verificables, McNemar da p = 0.11 con Mistral y 0.51 con Qwen. Lo consistente en las dos corridas es que el multi-agente entrega menos respuestas incorrectas.

**¿El código que ejecuta el sistema es seguro?**
Corre en un proceso aparte con límite de tiempo, CPU y memoria, pero sin bloquear red ni archivos. Es aceptable porque la VM de Colab es desechable; no lo pondríamos así en producción.

**¿Cuánto más lento es?**
Unos 58 segundos por consulta contra 14 del Naive RAG, sin contar al juez. Es el costo de las herramientas y de las rondas de verificación.
