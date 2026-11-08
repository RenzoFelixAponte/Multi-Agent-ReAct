"""Plantillas de prompt de los agentes. Solo texto: sin lógica ni dependencias."""

REACT_PROMPT = """Responde la pregunta usando el formato:
Thought: ...
Action: kb[consulta] o web[consulta]
Observation: (lo escribe el sistema)
... repite ...
Final: respuesta final

Pregunta: {question}
Retroalimentación previa del auditor: {feedback}
{scratch}"""

AUDIT_PROMPT = """Eres el Auditor de Hechos. Verifica si la respuesta está respaldada por la evidencia.
Responde solo "APROBADO" o "RECHAZADO: motivo".

Pregunta: {question}
Respuesta: {answer}
Evidencia:
{evidence}"""

REDACT_PROMPT = """Redacta una respuesta clara y citada en base a la respuesta verificada.
Pregunta: {question}
Respuesta verificada: {answer}
Fuentes: {sources}"""

NAIVE_PROMPT = """Responde usando solo el contexto.
Contexto:
{context}

Pregunta: {question}
Respuesta:"""
