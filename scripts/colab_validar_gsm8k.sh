#!/usr/bin/env bash
# Valida en una T4 de Colab el cambio del Auditor de GSM8K sobre problemas del split train.
#
# Uso:  scripts/colab_validar_gsm8k.sh               # código de 7f025ef contra HEAD, 20 problemas
#       BASE=<commit> N=30 scripts/colab_validar_gsm8k.sh
#
# Los resultados quedan en results/validacion_gsm8k/. Como colab_run.sh, baja lo avanzado cada
# SYNC_EVERY segundos, y al volver a ejecutarlo sube lo que ya hay y continúa desde ahí.
set -euo pipefail

cd "$(dirname "$0")/.."
SESSION="${SESSION:-validacion}"
BASE="${BASE:-7f025ef}"   # última versión con el Auditor de GSM8K que opina en texto libre
N="${N:-20}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}"
SYNC_EVERY="${SYNC_EVERY:-300}"
out="results/validacion_gsm8k"
files=(anterior.jsonl nueva.jsonl naive_rag.jsonl resumen.csv)

if ! git diff --quiet HEAD -- src scripts/validar_gsm8k.py; then
  echo "Hay cambios sin commit en src/: la versión nueva se toma de HEAD." >&2
  exit 1
fi
mkdir -p "$out"

lines() { if [[ -f "$1" ]]; then wc -l < "$1"; else echo 0; fi; }
fetch() {
  local tmp
  tmp="$(mktemp)"
  if colab download -s "$SESSION" "/content/results/validacion_gsm8k/$1" "$tmp" >/dev/null 2>&1 \
     && [[ -s "$tmp" ]] && { [[ "$1" != *.jsonl ]] || (( $(lines "$tmp") >= $(lines "$out/$1") )); }; then
    mv "$tmp" "$out/$1"
    return 0
  fi
  rm -f "$tmp"
  return 1
}

work="$(mktemp -d)"
archive="$(mktemp --suffix=.tar.gz)"
extract="$(mktemp --suffix=.py)"
sync_pid=""
trap 'rm -rf "$work" "$archive" "$extract"; [[ -n "$sync_pid" ]] && kill "$sync_pid" 2>/dev/null || true' EXIT

mkdir -p "$work/validacion/anterior" "$work/validacion/nueva" "$work/validacion/previos"
git archive "$BASE" src | tar x -C "$work/validacion/anterior"
git archive HEAD src | tar x -C "$work/validacion/nueva"
cp scripts/validar_gsm8k.py "$work/validacion/"
for f in "$out"/*.jsonl; do [[ -s "$f" ]] && cp "$f" "$work/validacion/previos/" && echo "Se reanuda con: $f"; done
tar czf "$archive" -C "$work" validacion

cat > "$extract" <<'PY'
import glob, os, shutil, subprocess
subprocess.run("tar xzf /content/validacion.tar.gz -C /content", shell=True, check=True)
os.makedirs("/content/results/validacion_gsm8k", exist_ok=True)
for path in glob.glob("/content/validacion/previos/*.jsonl"):
    shutil.copy(path, "/content/results/validacion_gsm8k/")
print("versiones:", sorted(os.listdir("/content/validacion")))
PY

echo "Valida $BASE contra $(git rev-parse --short HEAD) en $N problemas de GSM8K train con $MODEL_ID"
colab new -s "$SESSION" --gpu T4
colab upload -s "$SESSION" "$archive" /content/validacion.tar.gz
colab exec -s "$SESSION" -f "$extract"

(
  while sleep "$SYNC_EVERY"; do
    for f in anterior.jsonl nueva.jsonl naive_rag.jsonl; do fetch "$f" || true; done
    echo "[respaldo $(date +%H:%M)] anterior=$(lines "$out/anterior.jsonl") nueva=$(lines "$out/nueva.jsonl") naive=$(lines "$out/naive_rag.jsonl")"
  done
) &
sync_pid=$!

status=0
colab exec -s "$SESSION" --timeout 0 --env "N=$N" --env "MODEL_ID=$MODEL_ID" -f scripts/validar_gsm8k.py || status=$?
kill "$sync_pid" 2>/dev/null || true
sync_pid=""

for f in "${files[@]}"; do fetch "$f" || echo "sin $f nuevo en la VM"; done
if (( status != 0 )); then
  echo "La validación se interrumpió (código $status). Vuelve a ejecutar el mismo comando para reanudar." >&2
  exit "$status"
fi
colab stop -s "$SESSION"
