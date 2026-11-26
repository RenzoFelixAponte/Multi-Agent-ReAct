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
#
# Si Colab corta la VM, no se pierde lo avanzado: mientras corre, evaluacion.jsonl se baja cada
# SYNC_EVERY segundos (300 por defecto), y al volver a ejecutar el mismo comando los .jsonl locales
# se suben a la VM nueva y la evaluación salta lo que ya está hecho.
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
SYNC_EVERY="${SYNC_EVERY:-300}"
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
out="results/$RUN_NAME"
mkdir -p "$out"

lines() { if [[ -f "$1" ]]; then wc -l < "$1"; else echo 0; fi; }

# Baja un archivo de resultados sin pisar uno local más completo: si la VM murió o el archivo
# todavía no existe allá, el local queda como estaba.
fetch() {
  local tmp
  tmp="$(mktemp)"
  if colab download -s "$SESSION" "/content/results/$RUN_NAME/$1" "$tmp" >/dev/null 2>&1 \
     && [[ -s "$tmp" ]] && { [[ "$1" != *.jsonl ]] || (( $(lines "$tmp") >= $(lines "$out/$1") )); }; then
    mv "$tmp" "$out/$1"
    return 0
  fi
  rm -f "$tmp"
  return 1
}

fetch_all() {
  for f in prueba_tiempo.jsonl evaluacion.jsonl tabla_comparativa.csv resultados_por_consulta.csv; do
    fetch "$f" || echo "sin $f nuevo en la VM"
  done
}

archive="$(mktemp --suffix=.tar.gz)"
extract="$(mktemp --suffix=.py)"
sync_pid=""
trap 'rm -f "$archive" "$extract"; [[ -n "$sync_pid" ]] && kill "$sync_pid" 2>/dev/null || true' EXIT

# Lo avanzado en una corrida anterior viaja con el código y se reanuda desde ahí.
partial=()
for f in "$out"/prueba_tiempo.jsonl "$out"/evaluacion.jsonl; do
  if [[ -s "$f" ]]; then partial+=("$f"); fi
done
tar czf "$archive" src "${partial[@]}"
if (( ${#partial[@]} )); then echo "Se reanuda con: ${partial[*]}"; fi

# Con un archivo en vez de `echo '!cmd' | colab exec`, que una vez quedó colgado sin salida.
cat > "$extract" <<'PY'
import glob, os, shutil, subprocess
subprocess.run("mkdir -p /content/Multi-Agent-ReAct && tar xzf /content/src.tar.gz -C /content/Multi-Agent-ReAct",
               shell=True, check=True)
run = os.environ["RUN_NAME"]
target = f"/content/results/{run}"
os.makedirs(target, exist_ok=True)
for path in glob.glob(f"/content/Multi-Agent-ReAct/results/{run}/*.jsonl"):
    shutil.copy(path, target)
    print("resultados previos:", os.path.basename(path), sum(1 for _ in open(path, encoding="utf-8")), "registros")
print("código extraído en /content/Multi-Agent-ReAct")
PY

# Una sesión que quedó abierta de una corrida cortada: primero se rescata lo que tenga.
if colab sessions 2>/dev/null | grep -qw -- "$SESSION"; then
  echo "La sesión $SESSION sigue abierta: bajo sus resultados y la cierro antes de empezar."
  fetch_all
  colab stop -s "$SESSION" || true
fi

colab new -s "$SESSION" --gpu T4
colab upload -s "$SESSION" "$archive" /content/src.tar.gz
colab exec -s "$SESSION" --env "RUN_NAME=$RUN_NAME" -f "$extract"

# Respaldo periódico mientras corre el notebook.
(
  while sleep "$SYNC_EVERY"; do
    if fetch evaluacion.jsonl; then echo "[respaldo $(date +%H:%M)] $(lines "$out/evaluacion.jsonl") registros en $out"; fi
  done
) &
sync_pid=$!

status=0
colab exec -s "$SESSION" --timeout 0 "${envs[@]}" -f notebooks/proyecto2.ipynb || status=$?
kill "$sync_pid" 2>/dev/null || true
sync_pid=""

fetch_all
if (( status != 0 )); then
  echo "La corrida se interrumpió (código $status). Lo avanzado quedó en $out/ ($(lines "$out/evaluacion.jsonl") registros)." >&2
  echo "Vuelve a ejecutar el mismo comando para reanudar. Si la VM sigue viva, la sesión quedó abierta para inspeccionarla." >&2
  exit "$status"
fi
colab stop -s "$SESSION"
