"""Plantillas de prompt de los agentes. Solo texto: sin lógica ni dependencias."""

REACT_PROMPT = """Responde la pregunta alternando Thought, Action y Observation, y termina con Final.

Herramientas disponibles:
{tools}

Ejemplo:
Pregunta: ¿Quién escribió Cien años de soledad?
Thought: Necesito buscar el autor de esa novela.
Action: kb[autor de Cien años de soledad]
Observation: [Literatura] Gabriel García Márquez escribió Cien años de soledad en 1967.
Thought: La Observation ya responde la pregunta.
Final: Gabriel García Márquez escribió Cien años de soledad.

Reglas:
- Escribe una sola Action por turno y detente. La Observation la escribe el sistema, nunca tú.
- Dentro de los corchetes va el texto real que quieres buscar o calcular, no una palabra genérica como "consulta".
- No repitas una Action que ya hiciste. En cuanto las Observations (o la recuperación inicial) tengan lo necesario, escribe Final.
- Si es un problema numérico, calcula con calc y termina con "Final: Respuesta final: <número>".
- Si pide código, defínelo y pruébalo con assert dentro de un mismo python[...], y tras Final entrega el código completo en un bloque ```python.
- Si hay retroalimentación del auditor, corrige exactamente lo que señala.

Pregunta: {question}
Retroalimentación previa del auditor: {feedback}
{scratch}{last_step}"""

LAST_STEP_NOTICE = "Este es tu último paso: escribe Final ahora, con la mejor respuesta que respalden las Observations.\n"

TOOL_DESCRIPTIONS = {
    "kb": "kb[<qué buscar>]: busca fragmentos en la base de conocimiento local",
    "web": "web[<qué buscar>]: busca en la web",
    "calc": "calc[<expresión>]: evalúa aritmética, p. ej. calc[(16-3-4)*2]",
    "python": "python[<código>]: ejecuta código Python en un proceso nuevo y devuelve lo que imprime o el error. Cada ejecución es independiente: incluye en el mismo código las definiciones y las pruebas (assert)",
}

AUDIT_PROMPT = """Eres el Auditor de Hechos. Verifica si la respuesta está respaldada por la evidencia.
La evidencia puede incluir fuentes recuperadas, cálculos (calc) y ejecuciones de código (python):
un número obtenido de un cálculo, o un código que se ejecutó sin errores, cuenta como respaldado.
Si la pregunta pide código, la respuesta debe incluir el código completo en un bloque ```python y la evidencia
debe mostrar que ese código se ejecutó con sus pruebas; si no, rechaza.
Responde solo "APROBADO" o "RECHAZADO: motivo".

Pregunta: {question}
Respuesta: {answer}
Evidencia:
{evidence}"""

REDACT_PROMPT = """Redacta una respuesta clara y citada en base a la respuesta verificada.
Si la pregunta es un problema matemático, termina con una línea "Respuesta final: <número>".
Si la respuesta es código, entrégalo completo y sin cambios en un bloque ```python.
Pregunta: {question}
Respuesta verificada: {answer}
Fuentes: {sources}"""

NAIVE_PROMPT = """Responde la pregunta usando el contexto recuperado. Si el contexto no alcanza, razona tú mismo.
Si la pregunta es un problema matemático, termina con una línea "Respuesta final: <número>".
Si la pregunta pide código, entrégalo completo en un bloque ```python.
Contexto:
{context}

Pregunta: {question}
Respuesta:"""


FAITHFULNESS_PROMPT = """Evalúa la fidelidad de una respuesta a su contexto.
1. Descompón la respuesta en afirmaciones atómicas (máximo 8).
2. Para cada una indica si se puede inferir del contexto o de los datos que da la pregunta, sin conocimiento externo.
Formato obligatorio: una línea por afirmación que empiece exactamente con "SI:" o con "NO:", sin numerar y sin explicar.
Ejemplo:
SI: Lima es la capital del Perú
NO: Lima tiene diez millones de habitantes

Pregunta (sus datos cuentan como información dada): {question}

Contexto:
{context}

Respuesta:
{answer}"""

RELEVANCE_PROMPT = """Evalúa qué tan directamente responde la respuesta a la pregunta.
No juzgues si es verdadera, solo si atiende lo que se pregunta.
0 = no tiene relación, 5 = responde parcialmente o con relleno, 10 = responde de forma directa y completa.
Responde solo con una línea: PUNTAJE: <entero de 0 a 10>

Pregunta: {question}

Respuesta:
{answer}"""
