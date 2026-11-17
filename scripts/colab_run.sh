#!/usr/bin/env bash
# Corre notebooks/proyecto2.ipynb en una T4 de Colab con la CLI oficial y baja los resultados.
#
# Uso:  HF_TOKEN=hf_... scripts/colab_run.sh            # corrida completa
#       FULL_RUN=0 HF_TOKEN=hf_... scripts/colab_run.sh # solo la prueba de tiempo
#
# El primer comando de `colab` abre el login OAuth en el navegador.
set -euo pipefail

SESSION="${SESSION:-proyecto}"
FULL_RUN="${FULL_RUN:-1}"
: "${HF_TOKEN:?Define HF_TOKEN (Mistral-7B-Instruct-v0.3 es un modelo restringido en Hugging Face)}"

cd "$(dirname "$0")/.."
mkdir -p results

archive="$(mktemp --suffix=.tar.gz)"
trap 'rm -f "$archive"' EXIT
tar czf "$archive" src

colab new -s "$SESSION" --gpu T4
# Si algo falla, la sesión queda abierta para inspeccionarla: cerrar con `colab stop -s $SESSION`.
colab upload -s "$SESSION" "$archive" /content/src.tar.gz
echo '!mkdir -p /content/Multi-Agent-ReAct && tar xzf /content/src.tar.gz -C /content/Multi-Agent-ReAct' \
  | colab exec -s "$SESSION"

colab exec -s "$SESSION" --timeout 0 --env "HF_TOKEN=$HF_TOKEN" --env "FULL_RUN=$FULL_RUN" \
  -f notebooks/proyecto2.ipynb

for f in prueba_tiempo.jsonl evaluacion.jsonl tabla_comparativa.csv resultados_por_consulta.csv; do
  colab download -s "$SESSION" "/content/results/$f" "results/$f" || echo "sin $f (¿FULL_RUN=0?)"
done
colab stop -s "$SESSION"
