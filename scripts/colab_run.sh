#!/usr/bin/env bash
# Corre notebooks/proyecto2.ipynb en una T4 de Colab con la CLI oficial y baja los resultados.
#
# Uso:  scripts/colab_run.sh                    # corrida completa: Qwen2.5-7B y juez de NVIDIA
#       FULL_RUN=0 scripts/colab_run.sh         # solo la prueba de tiempo
#       MODEL_ID=mistralai/Mistral-7B-Instruct-v0.3 JUDGE_BACKEND=same scripts/colab_run.sh
#
# NVIDIA_API_KEY (juez), NVIDIA_MODEL y HF_TOKEN se toman del entorno o, si no, del .env de la raíz.
# Los resultados quedan en results/<RUN_NAME>/ (por defecto, el nombre del modelo en minúsculas).
# El primer comando de `colab` abre el login OAuth en el navegador.
set -euo pipefail

cd "$(dirname "$0")/.."

# Lee una variable del .env sin ejecutarlo como script.
env_file_value() {
  [[ -f .env ]] || return 0
  sed -n "s/^$1=//p" .env | tail -n 1 | sed -e 's/^["'\'']//' -e 's/["'\'']$//'
}
NVIDIA_API_KEY="${NVIDIA_API_KEY:-$(env_file_value NVIDIA_API_KEY)}"
NVIDIA_MODEL="${NVIDIA_MODEL:-$(env_file_value NVIDIA_MODEL)}"
HF_TOKEN="${HF_TOKEN:-$(env_file_value HF_TOKEN)}"

SESSION="${SESSION:-proyecto}"
FULL_RUN="${FULL_RUN:-1}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}"
JUDGE_BACKEND="${JUDGE_BACKEND:-nvidia}"
RUN_NAME="${RUN_NAME:-$(basename "$MODEL_ID" | tr '[:upper:]' '[:lower:]')}"

if [[ "$JUDGE_BACKEND" == nvidia && -z "$NVIDIA_API_KEY" ]]; then
  echo "Falta NVIDIA_API_KEY para el juez remoto: ponla en .env o usa JUDGE_BACKEND=same." >&2
  exit 1
fi

envs=(--env "FULL_RUN=$FULL_RUN" --env "MODEL_ID=$MODEL_ID" --env "JUDGE_BACKEND=$JUDGE_BACKEND"
      --env "RUN_NAME=$RUN_NAME")
if [[ -n "$NVIDIA_API_KEY" ]]; then envs+=(--env "NVIDIA_API_KEY=$NVIDIA_API_KEY"); fi
if [[ -n "$NVIDIA_MODEL" ]]; then envs+=(--env "NVIDIA_MODEL=$NVIDIA_MODEL"); fi
if [[ -n "$HF_TOKEN" ]]; then envs+=(--env "HF_TOKEN=$HF_TOKEN"); fi

echo "Modelo: $MODEL_ID | juez: $JUDGE_BACKEND | resultados: results/$RUN_NAME | FULL_RUN=$FULL_RUN"
mkdir -p "results/$RUN_NAME"

archive="$(mktemp --suffix=.tar.gz)"
extract="$(mktemp --suffix=.py)"
trap 'rm -f "$archive" "$extract"' EXIT
tar czf "$archive" src
# Con un archivo en vez de `echo '!cmd' | colab exec`, que una vez quedó colgado sin salida.
cat > "$extract" <<'PY'
import subprocess
subprocess.run("mkdir -p /content/Multi-Agent-ReAct && tar xzf /content/src.tar.gz -C /content/Multi-Agent-ReAct",
               shell=True, check=True)
print("código extraído en /content/Multi-Agent-ReAct")
PY

colab new -s "$SESSION" --gpu T4
# Si algo falla, la sesión queda abierta para inspeccionarla: cerrar con `colab stop -s $SESSION`.
colab upload -s "$SESSION" "$archive" /content/src.tar.gz
colab exec -s "$SESSION" -f "$extract"

colab exec -s "$SESSION" --timeout 0 "${envs[@]}" -f notebooks/proyecto2.ipynb

for f in prueba_tiempo.jsonl evaluacion.jsonl tabla_comparativa.csv resultados_por_consulta.csv; do
  colab download -s "$SESSION" "/content/results/$RUN_NAME/$f" "results/$RUN_NAME/$f" || echo "sin $f (¿FULL_RUN=0?)"
done
colab stop -s "$SESSION"
