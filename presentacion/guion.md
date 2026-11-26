# Guion de la presentación

10 diapositivas, unos 10 minutos: 1 minuto por diapositiva en promedio. El reparto entre los tres integrantes es una propuesta; si lo cambian, cada uno debería quedarse con un bloque seguido para no pasarse la palabra a cada rato.

| Bloque | Diapositivas | Quién | Tiempo |
|---|---|---|---|
| El sistema | 1 a 4 | Renzo | 3 min 30 s |
| La evaluación | 5 a 7 | Jose | 3 min |
| Hallazgos y cierre | 8 a 10 | Diego | 3 min 30 s |

Todos los números de las diapositivas salen de `results/` (ver el encabezado de `presentacion.tex`). Si alguien pregunta por uno, al final hay respuestas preparadas.

---

## 1. Portada (Renzo, 20 s)

> Buenas. Somos Renzo, Jose y Diego, y les vamos a presentar un sistema multi-agente ReAct: tres agentes que buscan, verifican y redactan, y que prefieren no responder antes que entregar una respuesta sin respaldo. El fondo de la portada son trazas reales del sistema razonando.

## 2. El reto y nuestra arquitectura (Renzo, 1 min 10 s)

> El reto era usar un modelo chico, de 7 mil millones de parámetros y cuantizado a 4 bits en una Tesla T4, y aun así no propagar alucinaciones.
>
> Nuestra respuesta es un grafo cíclico en LangGraph con tres agentes. El **Investigador** razona con el ciclo Thought, Action, Observation y usa herramientas. El **Auditor de Hechos** revisa su respuesta: si la aprueba, pasa al **Redactor**, que solo la reescribe en limpio; si la rechaza, el flujo vuelve al Investigador con el motivo del rechazo. Después de tres rechazos, el sistema se abstiene.
>
> Antes del bucle, la base vectorial y la web se consultan en paralelo: eso es la concurrencia que pedía el enunciado.

## 3. El bucle ReAct y un rechazo real (Renzo, 1 min 10 s)

> Este es un caso real de la corrida, el problema 11 de MBPP: una función que quite la primera y la última aparición de un carácter, con tres pruebas.
>
> En la primera ronda el Investigador entrega una función. El Auditor no le pregunta a un LLM si el código "parece" correcto: lo **ejecuta** con las pruebas del enunciado, y la segunda prueba falla con un AssertionError.
>
> Ese error vuelve al Investigador. En la segunda ronda, su Thought dice que el problema era usar `replace`, cambia a `find` y `rfind`, ejecuta el código con las tres pruebas y la Observation dice "ok". Un detalle importante: la Observation la escribe el sistema, nunca el modelo. Ahora el Auditor aprueba y el Redactor entrega el código sin tocarlo.

## 4. Un grafo por tarea (Renzo, 50 s)

> Usamos tres datasets y cada uno tiene su grafo. En las preguntas de WikiQA, el Investigador usa la base vectorial y la web, y el Auditor contrasta la respuesta con la evidencia. En matemáticas de GSM8K usa una calculadora segura, y en código de MBPP, Python en un proceso aislado.
>
> Tres decisiones: en ChromaDB indexamos también las frases incorrectas de WikiQA, para que el Naive RAG no reciba la respuesta servida. MBPP admite cinco rondas, porque ahí el Auditor verifica ejecutando y reintentar no puede dejar pasar código erróneo. Y en matemáticas y código no hay web, porque podría devolver la solución publicada del benchmark.

## 5. Cómo evaluamos (Jose, 1 min)

> Evaluamos sobre 50 consultas: 20 de WikiQA, 15 de GSM8K y 15 de MBPP. Las mismas consultas para los dos métodos.
>
> La línea base es un **Naive RAG**: una recuperación de tres fragmentos y una sola llamada al modelo, sin herramientas ni auditoría. Los dos métodos usan el mismo Qwen2.5-7B.
>
> Medimos las dos métricas de RAGAS que pide el enunciado. **Faithfulness** es la fracción de afirmaciones de la respuesta que se pueden inferir del contexto. **Answer Relevance** es qué tan directamente responde a la pregunta. En GSM8K y MBPP además verificamos de forma objetiva: el número final y las pruebas ejecutadas. El juez es un modelo independiente de 120B, el mismo para los dos métodos. Una abstención puntúa cero.

## 6. Resultados sobre las 50 consultas (Jose, 1 min)

> El multi-agente gana en **fidelidad**, 0.80 contra 0.68, y entrega menos respuestas incorrectas: 10 % contra 17 %.
>
> Pierde en Answer Relevance y en exactitud. La razón está a la derecha: si comparamos solo las 44 consultas que los dos responden, el multi-agente gana también en relevancia, 0.977 contra 0.973, y en fidelidad la diferencia sube a 0.91 contra 0.70. Lo que baja sus promedios son sus 6 abstenciones, que puntúan cero. Y la exactitud la pierde por un solo dataset, que es lo que vemos ahora.

## 7. Dónde gana y dónde pierde (Jose, 1 min)

> En **WikiQA** gana en fidelidad, 0.84 contra 0.66: responde con lo que recuperó.
>
> En **MBPP** gana en exactitud, 13 contra 11 de 15, y sobre todo no entregó ni un código incorrecto; el Naive RAG entregó 4.
>
> En **GSM8K** pierde: 9 aciertos contra 14. Revisamos los 6 fallos uno por uno: en 3 el Investigador tenía la respuesta correcta y el Auditor la rechazó. O sea, el problema no era el Investigador, era cómo verificaba el Auditor.

## 8. El juez importa tanto como el modelo (Diego, 1 min 10 s)

> Nuestra primera corrida fue con Mistral-7B, y el juez era el mismo Mistral. Ahí el multi-agente perdía en las dos métricas del enunciado.
>
> Al revisar los datos vimos que ese juez de 7B no medía bien: les daba **más** fidelidad a las respuestas incorrectas del multi-agente, 0.86, que a las correctas, 0.73. El juez de 120B sí separa: 0.36 contra 0.78.
>
> Por eso cambiamos dos cosas. El orquestador pasó a Qwen2.5-7B, que sigue siendo de 7B y local, como exige el enunciado. Y el juez pasó a ser independiente: el enunciado obliga a que el orquestador sea local, pero no el juez.

## 9. Un guardrail vale lo que su verificación (Diego, 1 min 10 s)

> Esta es la lección principal del proyecto. Comparen los dos Auditores.
>
> En MBPP, el Auditor **ejecuta** el código: no entregó ningún código incorrecto con ninguno de los dos modelos. Cuando rechaza, el error de ejecución vuelve al Investigador: pasó en 3 consultas, corrigió una, y en las otras dos se abstuvo en lugar de entregar código roto (el Naive RAG también falló ahí).
>
> En GSM8K, el Auditor **opinaba** en texto libre. Miren este caso: la respuesta correcta era 3. El Auditor escribe que el total es 2 más 1 igual a 3, que eso "es correcto"... y la rechaza. Tres veces, así que el sistema se abstuvo.
>
> Lo cambiamos: ahora el Auditor resuelve el problema por su cuenta, sin ver la respuesta, y solo rechaza si los números no coinciden. Para no ajustarlo a nuestras 15 preguntas, lo validamos en 20 problemas que no están en la evaluación.

## 10. Lecciones aprendidas (Diego, 50 s)

> Cuatro lecciones. Una: verificar ejecutando, no opinando. Dos: un juez débil no distingue una respuesta correcta de una incorrecta, así que hay que medir con uno independiente. Tres: abstenerse baja las métricas promediadas, pero evita entregar errores, que es justamente lo que pedía el enunciado. Y cuatro, un detalle práctico: el 7B respondía en español a preguntas en inglés hasta que le nombramos el idioma.
>
> Gracias. ¿Preguntas?

---

## Preguntas probables

**¿Por qué el multi-agente pierde en Answer Relevance si es "mejor"?**
Por las abstenciones. Una abstención puntúa 0. En las 44 consultas que responden los dos, el multi-agente gana: 0.977 contra 0.973. Elegimos no esconder las abstenciones: la tabla las cuenta y reporta aparte la fracción de consultas respondidas.

**¿No es trampa cambiar el juez después de ver los resultados?**
El cambio fue porque el juez de 7B no distinguía respuestas correctas de incorrectas: lo comprobamos con la verificación objetiva de GSM8K y MBPP. El juez nuevo es el mismo para los dos métodos, y los resultados con Mistral siguen en el repositorio.

**¿Y el cambio del Auditor de GSM8K no está ajustado a las 15 preguntas?**
Lo validamos en 20 problemas del split *train* de GSM8K, que no están en la evaluación (que usa el split *test*), comparando el código anterior con el nuevo (`results/validacion_gsm8k/`).

**¿Por qué el Naive RAG resuelve tan bien GSM8K?**
Porque Qwen2.5-7B razonando paso a paso ya es muy bueno en GSM8K, y la base de WikiQA no le estorba. Con Mistral era al revés: el Naive RAG acertaba 0.30 y el multi-agente 0.50.

**¿Es significativa la diferencia en exactitud?**
No. Con 30 consultas verificables, McNemar da p = 0.51: el multi-agente acierta 3 que el Naive RAG falla y el Naive RAG acierta 6 que el multi-agente falla. Lo consistente en las dos corridas es que el multi-agente entrega menos respuestas incorrectas.

**¿El código que ejecuta el sistema es seguro?**
Corre en un proceso aparte con límite de tiempo, CPU y memoria, pero sin bloquear red ni archivos. Es aceptable porque la VM de Colab es desechable; no lo pondríamos así en producción.

**¿Cuánto más lento es?**
Unos 58 segundos por consulta contra 14 del Naive RAG, sin contar al juez: es el costo de las herramientas y las rondas de verificación.
