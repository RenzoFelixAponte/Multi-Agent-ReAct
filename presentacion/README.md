# Presentación

Beamer, 16:10, 10 diapositivas. `main.pdf` es la última versión compilada y `guion.md` es lo que se dice en cada una.

Combina las dos versiones que hizo el grupo: la de 18 diapositivas con tema propio (commit `dea820f`) y la de 10 con la plantilla de la presentación de Computer Vision. Quedó la plantilla de Computer Vision, con 10 diapositivas como máximo. De la primera versión vienen los requisitos del enunciado, lo que corregimos al probar en la T4, las cifras del juez, las limitaciones y la lección de que más andamiaje no siempre ayuda.

- `main.tex`: el contenido, una diapositiva por `\begin{frame}`.
- `estilo/tema.tex`: el aspecto de todas las diapositivas salvo la portada (colores, letra, pie con la navegación).
- `estilo/portada-slate.tex`: la portada, adaptada de slate-blocks (CC BY 4.0, ver `estilo/LICENSE-slate-blocks`).
- `figuras/gen/portada_fondo.pdf`: el fondo de la portada, con trazas ReAct reales de la corrida. Lo genera `scripts/portada_fondo.py`.
- `guion.md`: el texto de cada diapositiva, el reparto entre los integrantes y respuestas a preguntas probables.

No hay imágenes externas salvo el logo: el grafo de agentes está dibujado con TikZ y los gráficos con pgfplots.

## Compilar

- **tectonic** (descarga los paquetes que falten): `tectonic -X compile main.tex`
- **TeX Live o MacTeX:** `latexmk -pdf main.tex` (dos pasadas: la portada usa `remember picture`).
- **Overleaf:** sube la carpeta `presentacion/` completa y compila `main.tex`.

Para regenerar el fondo de la portada:

```bash
python presentacion/scripts/portada_fondo.py
cd presentacion/figuras/gen && tectonic portada_fondo.tex
```

## Orden de la charla (unos 10 minutos)

| Parte | Diapositivas | Qué se cuenta |
|---|---|---|
| Contexto | 1–2 | El problema y lo que pedía el enunciado |
| El sistema | 3–5 | El grafo, el ciclo ReAct con un rechazo real, las herramientas y lo que corregimos en la T4 |
| Evaluación | 6–7 | Las métricas y los resultados |
| Hallazgos y cierre | 8–10 | El juez, el Auditor, lo que aprendimos y las limitaciones |

Las cifras salen de `results/evaluacion.jsonl` (Mistral-7B), de `results/qwen2.5-7b-instruct*/evaluacion.jsonl` (Qwen2.5-7B) y de `results/validacion_gsm8k/`. Para recalcularlas, usa `python scripts/analisis_errores.py <archivo>` y `python scripts/resumir_resultados.py <archivo> <tabla.csv>`.
