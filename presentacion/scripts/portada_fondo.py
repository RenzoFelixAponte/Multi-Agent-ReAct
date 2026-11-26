"""Genera el fondo de la portada con trazas ReAct reales de la corrida con Qwen2.5-7B.

Toma las trazas guardadas en results/qwen2.5-7b-instruct/evaluacion.jsonl, las escapa para LaTeX,
resalta Thought, Action, Observation y Final, y escribe figuras/gen/portada_fondo.tex, que luego se
compila a PDF (16:10, como las diapositivas).

Uso:  python presentacion/scripts/portada_fondo.py
      cd presentacion/figuras/gen && tectonic portada_fondo.tex
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results" / "qwen2.5-7b-instruct" / "evaluacion.jsonl"
TARGET = ROOT / "presentacion" / "figuras" / "gen" / "portada_fondo.tex"
# una consulta de cada dataset, todas resueltas por el multi-agente
PICKS = [("mbpp", "11"), ("wiki_qa", "Q11"), ("gsm8k", "0"), ("mbpp", "18")]
MAX_LINES = 78
MAX_CHARS = 118

_ESCAPES = {"\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "#": r"\#", "$": r"\$", "%": r"\%",
            "&": r"\&", "_": r"\_", "^": r"\^{}", "~": r"\~{}"}
KEYWORDS = ("Thought:", "Action:", "Observation:", "Final:")


def escape(text: str) -> str:
    return "".join(_ESCAPES.get(ch, ch) for ch in text)


def render(line: str) -> str:
    line = line.rstrip()[:MAX_CHARS]
    for kw in KEYWORDS:
        if line.startswith(kw):
            return r"{\color{cAcc}\bfseries " + kw + "}" + escape(line[len(kw):])
    return escape(line).replace(" ", "~") if line.startswith(" ") else escape(line)


def main() -> None:
    records = {}
    for line in SOURCE.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["method"] == "multiagente":
            records[(r["dataset"], r["item_id"])] = r
    lines: list[str] = []
    for key in PICKS:
        trace = records[key]["trace"]
        lines += [l for l in trace.splitlines() if l.strip()]
    # `\\{}`: si una línea empieza con [, `\\[...]` se leería como un espacio vertical
    body = "\\\\{}\n".join(render(l) for l in lines[:MAX_LINES])
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(r"""% Generado por presentacion/scripts/portada_fondo.py: trazas ReAct reales de la corrida.
\documentclass[border=0pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage[scale=0.86]{FiraMono}
\usepackage{xcolor}
\usepackage{tikz}
\definecolor{cAcc}{HTML}{4F5FC8}
\definecolor{cInk}{HTML}{2E3140}
\begin{document}
\begin{tikzpicture}
  \clip (0,0) rectangle (240mm,150mm);
  \fill[white] (0,0) rectangle (240mm,150mm);
  \node[anchor=north west, inner sep=6mm, text width=228mm, align=left, text=cInk!70,
        font=\ttfamily\fontsize{6.2}{7.7}\selectfont] at (0,150mm) {%
""" + body + r"""};
\end{tikzpicture}
\end{document}
""", encoding="utf-8")
    print(f"{min(len(lines), MAX_LINES)} líneas -> {TARGET}")


if __name__ == "__main__":
    main()
