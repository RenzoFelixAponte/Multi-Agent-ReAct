# Presentación

Beamer, 16:9, 18 diapositivas. `main.pdf` es la última versión compilada.

- `main.tex`: el contenido. Una diapositiva por `\begin{frame}`, cada una con sus notas del orador en `\note{...}`.
- `tema.tex`: el aspecto (colores, letras, encabezados, tarjetas). Para cambiar un color o un tamaño en todas las diapositivas, se edita aquí.

No hay imágenes externas: el grafo de agentes está dibujado con TikZ y los gráficos con pgfplots.

## Compilar

- **Overleaf:** sube la carpeta `presentacion/` y compila `main.tex`. Funciona con pdfLaTeX y con XeLaTeX.
- **tectonic:** `tectonic -X compile main.tex`
- **TeX Live o MacTeX:** `latexmk -pdf main.tex`

## Presentar con notas

En `main.tex`, quita el `%` de la línea `% \setbeameroption{show notes on second screen=right}` y vuelve a compilar. Cada página queda con la diapositiva a la izquierda y sus notas a la derecha. En un visor de presentaciones como [pdfpc](https://pdfpc.github.io/), el público ve solo la diapositiva y tú ves las notas.

## Orden de la charla (unos 15 minutos)

| Parte | Diapositivas | Qué se cuenta |
|---|---|---|
| Contexto | 1–4 | El problema, lo que pedía el enunciado y los datos |
| El sistema | 5–9 | El grafo, el ciclo ReAct con una traza real, el Auditor, las herramientas y lo que corregimos en la T4 |
| Evaluación | 10–15 | Las métricas, las dos corridas, los resultados por dataset y un caso real en que el Auditor falla |
| Cierre | 16–18 | Lecciones, limitaciones y preguntas |

Las cifras salen de `results/evaluacion.jsonl` (Mistral-7B) y de `results/qwen2.5-7b-instruct/evaluacion.jsonl` (Qwen2.5-7B). Para recalcularlas, usa `python scripts/analisis_errores.py <archivo>`.
