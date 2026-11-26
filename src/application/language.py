"""Idioma de la pregunta, para pedir la respuesta en ese idioma de forma explícita.

Los prompts están en español y un 7B tiende a contestar en español aunque la pregunta esté en
inglés: en la corrida con Mistral, 9 de las 20 respuestas de WikiQA salieron en español a pesar de
la regla "en el idioma de la pregunta". Nombrar el idioma es más fuerte que pedir que lo deduzca.
"""
from __future__ import annotations

import re

_WORD = re.compile(r"[a-záéíóúüñ]+")
_SPANISH = frozenset(
    "el la los las del que y en un una es son por para con se su sus lo como más pero qué cuál "
    "cuáles cuánto cuántos cuántas quién dónde cuándo hay tiene este esta".split()
)
_ENGLISH = frozenset(
    "the of and to in is are was were for on with that by from as at an be this which what how "
    "who when where does did do many much".split()
)
UNKNOWN = "el idioma de la pregunta"


def question_language(question: str) -> str:
    """'inglés', 'español' o, si no se puede decidir, `UNKNOWN`.

    Solo mira la primera línea, que es el enunciado: en MBPP la pregunta sigue con una instrucción
    en español y los `assert`, que no dicen nada del idioma del enunciado.
    """
    first = next((line for line in question.splitlines() if line.strip()), "")
    words = _WORD.findall(first.lower())
    spanish = sum(w in _SPANISH for w in words)
    english = sum(w in _ENGLISH for w in words)
    if english > spanish:
        return "inglés"
    if spanish > english:
        return "español"
    return UNKNOWN
