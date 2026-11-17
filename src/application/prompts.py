"""Plantillas de prompt de los agentes. Solo texto: sin lógica ni dependencias."""

REACT_PROMPT = """Responde la pregunta usando el formato:
Thought: razonamiento breve
Action: herramienta[argumento]
Observation: (lo escribe el sistema)
... repite Thought/Action/Observation las veces necesarias ...
Final: respuesta final

Herramientas disponibles:
{tools}

Reglas:
- Escribe una sola Action por turno y detente: no escribas Observation, la escribe el sistema.
- Si la pregunta es un problema numérico, calcula con calc y termina con "Final: Respuesta final: <número>".
- Si la pregunta pide código, pruébalo con python (usa print y assert) y entrega el código completo en un bloque ```python tras Final:.
- Si hay retroalimentación del auditor, corrige exactamente lo que señala.

Pregunta: {question}
Retroalimentación previa del auditor: {feedback}
{scratch}"""

TOOL_DESCRIPTIONS = {
    "kb": "kb[consulta]: busca fragmentos en la base de conocimiento local",
    "web": "web[consulta]: busca en la web",
    "calc": "calc[expresión]: evalúa aritmética, p. ej. calc[(16-3-4)*2]",
    "python": "python[código]: ejecuta código Python y devuelve lo que imprime o el error",
}

AUDIT_PROMPT = """Eres el Auditor de Hechos. Verifica si la respuesta está respaldada por la evidencia.
La evidencia puede incluir fuentes recuperadas, cálculos (calc) y ejecuciones de código (python):
un número obtenido de un cálculo, o un código que se ejecutó sin errores, cuenta como respaldado.
Responde solo "APROBADO" o "RECHAZADO: motivo".

Pregunta: {question}
Respuesta: {answer}
Evidencia:
{evidence}"""

REDACT_PROMPT = """Redacta una respuesta clara y citada en base a la respuesta verificada.
Si la respuesta es un número, termina con una línea "Respuesta final: <número>".
Si la respuesta es código, entrégalo completo y sin cambios en un bloque ```python.
Pregunta: {question}
Respuesta verificada: {answer}
Fuentes: {sources}"""

NAIVE_PROMPT = """Responde la pregunta usando el contexto recuperado. Si el contexto no alcanza, razona tú mismo.
Si la respuesta es un número, termina con una línea "Respuesta final: <número>".
Si la pregunta pide código, entrégalo completo en un bloque ```python.
Contexto:
{context}

Pregunta: {question}
Respuesta:"""
