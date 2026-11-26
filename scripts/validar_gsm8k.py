"""Valida el cambio del Auditor de GSM8K en problemas que no están en el set de evaluación.

Compara dos versiones del código (`anterior` y `nueva`) sobre los primeros N problemas del split
*train* de GSM8K: el set de evaluación usa el split *test*, así que el cambio no se ajusta a esas
15 preguntas. También corre el Naive RAG una vez, como referencia. Solo mide la verificación
objetiva (número final), sin juez.

Se ejecuta en una VM de Colab con `scripts/colab_validar_gsm8k.sh`, que sube cada versión del
código a /content/validacion/<versión>/src. Cada versión corre en su propio proceso, para que sus
módulos no se mezclen. Los resultados se guardan uno por línea y una corrida cortada se reanuda.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

BASE = Path("/content/validacion")
OUT = Path("/content/results/validacion_gsm8k")
VARIANTS = ("anterior", "nueva")


def _done(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    ids = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            ids.add(json.loads(line)["item_id"])
        except (json.JSONDecodeError, KeyError):
            continue  # línea cortada: se recalcula
    return ids


def _append(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def worker(variant: str, n: int, model_id: str, with_naive: bool) -> None:
    sys.path.insert(0, str(BASE / variant / "src"))
    from adapters.fake_search import StaticSearch
    from adapters.hf_dataset_adapter import HFDatasetAdapter
    from adapters.in_memory_kb import InMemoryKB
    from adapters.local_llm import LocalLLM
    from application.answer_checks import numbers_match
    from application.graph import build_flow

    items = HFDatasetAdapter().load("gsm8k", "train", n)
    llm = LocalLLM(model_id)
    # GSM8K solo usa calc: la web y la base no se consultan.
    flow = build_flow(llm, StaticSearch([]), InMemoryKB([]), max_rounds=3, task="math")
    path = OUT / f"{variant}.jsonl"
    done = _done(path)
    for item in items:
        if item.item_id in done:
            continue
        start = time.perf_counter()
        state = flow.invoke({"question": item.question})
        answer = state.get("final", "")
        record = dict(variant=variant, item_id=item.item_id, answered=bool(answer.strip()),
                      correct=bool(answer.strip()) and numbers_match(answer, item.reference),
                      rounds=state.get("rounds", 1), rejections=state.get("rejections", []),
                      seconds=round(time.perf_counter() - start, 2), answer=answer, reference=item.reference)
        _append(path, record)
        print(f"[{variant}] {item.item_id:>3} respondida={record['answered']!s:5} correcta={record['correct']!s:5} "
              f"rondas={record['rounds']} {record['seconds']:6.1f}s", flush=True)

    if with_naive:  # el mismo Naive RAG de la evaluación: misma base de WikiQA y mismo prompt
        from adapters.chroma_kb import ChromaKB
        from adapters.hf_mappers import wikiqa_documents
        from application.naive_rag import naive_rag

        kb = ChromaKB(path="/content/chroma_db_validacion")
        kb.index(wikiqa_documents(HFDatasetAdapter().load("wiki_qa", "validation", limit=300)))
        path = OUT / "naive_rag.jsonl"
        done = _done(path)
        for item in items:
            if item.item_id in done:
                continue
            start = time.perf_counter()
            answer = naive_rag(item.question, llm, kb)
            record = dict(variant="naive_rag", item_id=item.item_id, answered=True,
                          correct=numbers_match(answer, item.reference), rounds=1, rejections=[],
                          seconds=round(time.perf_counter() - start, 2), answer=answer, reference=item.reference)
            _append(path, record)
            print(f"[naive_rag] {item.item_id:>3} correcta={record['correct']!s:5} {record['seconds']:6.1f}s", flush=True)


def summary() -> None:
    rows = []
    for name in (*VARIANTS, "naive_rag"):
        path = OUT / f"{name}.jsonl"
        records = {}
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                records[r["item_id"]] = r
        rs = list(records.values())
        if not rs:
            continue
        answered = [r for r in rs if r["answered"]]
        rows.append(dict(version=name, n=len(rs), respondidas=round(len(answered) / len(rs), 3),
                         exactitud=round(sum(r["correct"] for r in rs) / len(rs), 3),
                         incorrectas_entregadas=round(sum(r["answered"] and not r["correct"] for r in rs) / len(rs), 3),
                         rondas=round(sum(r["rounds"] for r in rs) / len(rs), 2),
                         seg_por_consulta=round(sum(r["seconds"] for r in rs) / len(rs), 1)))
    with (OUT / "resumen.csv").open("w", encoding="utf-8") as f:
        f.write(",".join(rows[0]) + "\n")
        for row in rows:
            f.write(",".join(str(v) for v in row.values()) + "\n")
            print(row, flush=True)


def orchestrate() -> None:
    n = int(os.environ.get("N", "20"))
    model_id = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "langgraph", "chromadb", "ddgs", "datasets",
                    "bitsandbytes", "accelerate"], check=True)
    OUT.mkdir(parents=True, exist_ok=True)
    for variant in VARIANTS:
        # Un proceso por versión: los paquetes `application` y `adapters` de cada una no se mezclan.
        cmd = [sys.executable, "-u", str(BASE / "validar_gsm8k.py"), "worker", variant, str(n), model_id,
               "1" if variant == "nueva" else "0"]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            if "it/s]" not in line and "%|" not in line:  # sin barras de progreso
                print(line, end="", flush=True)
        if proc.wait() != 0:
            raise RuntimeError(f"la versión {variant} terminó con código {proc.returncode}")
    summary()


if len(sys.argv) > 1 and sys.argv[1] == "worker":
    worker(sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5] == "1")
else:  # `colab exec -f` ejecuta el archivo dentro del kernel, con los argumentos del kernel
    orchestrate()
