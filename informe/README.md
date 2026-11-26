# Informe final

`main.tex` tiene la portada y los paquetes. El contenido está en `secciones/`, un archivo por sección, para que cada integrante pueda editar la suya sin conflictos. `main.pdf` es la última versión compilada.

La portada (título, integrantes y curso) está en `main.tex`.

## Compilar

Funciona con pdfLaTeX, XeLaTeX, LuaLaTeX y tectonic. Las figuras están hechas con TikZ y pgfplots, así que no hay imágenes externas, y la bibliografía está dentro del documento, así que no hace falta BibTeX.

- **Overleaf:** sube la carpeta `informe/` completa (o un `.zip` de ella) y compila `main.tex` con el compilador por defecto.
- **tectonic** (descarga los paquetes que falten): `tectonic -X compile main.tex`
- **TeX Live o MacTeX:** `latexmk -pdf main.tex`. Hay que compilar dos veces si usas `pdflatex` directo, para que las referencias cruzadas se resuelvan.

## Si cambian los resultados

Las cifras de las tablas salen de `results/tabla_comparativa.csv` y de `python scripts/analisis_errores.py`. Si se hace una corrida nueva, regenera la tabla con `python scripts/resumir_resultados.py` y actualiza `secciones/06-resultados.tex` y `secciones/07-analisis.tex`.
