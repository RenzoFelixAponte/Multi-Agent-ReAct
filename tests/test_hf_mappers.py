from proyecto2.adapters.hf_datasets import map_gsm8k, map_mbpp, map_rows, map_wikiqa, wikiqa_documents
from proyecto2.domain.models import Evidence


def test_wikiqa_keeps_only_positive_candidates():
    rows = [
        {"question_id": "Q1", "question": "capital del Perú?", "document_title": "Perú",
         "answer": "Lima es la capital.", "label": "1"},
        {"question_id": "Q1", "question": "capital del Perú?", "document_title": "Perú",
         "answer": "El Perú tiene muchos ríos.", "label": "0"},
    ]
    items = map_rows("wiki_qa", rows, limit=10)
    assert len(items) == 1
    assert items[0].reference == "Lima es la capital."
    assert items[0].context == (Evidence("Perú", "Lima es la capital."),)


def test_wikiqa_documents_come_from_context():
    items = map_rows("wiki_qa", [
        {"question_id": "Q1", "question": "q", "document_title": "T", "answer": "a", "label": 1},
    ], limit=5)
    assert wikiqa_documents(items) == [Evidence("T", "a")]


def test_gsm8k_extracts_final_number_and_drops_rows_without_marker():
    rows = [
        {"question": "Janet tiene 16 huevos...", "answer": "16 - 3 - 4 = <<16-3-4=9>>9\n#### 18"},
        {"question": "sin marca", "answer": "solo texto"},
        {"question": "miles", "answer": "razonamiento\n#### 1,200"},
    ]
    items = map_rows("gsm8k", rows, limit=10)
    assert [i.reference for i in items] == ["18", "1200"]


def test_mbpp_uses_text_as_question_and_code_as_reference():
    row = {"task_id": 11, "text": "Remove first occurrence", "code": "def f(s): return s"}
    item = map_mbpp(row, 0)
    assert item.item_id == "11"
    assert item.question == "Remove first occurrence"
    assert item.reference == "def f(s): return s"


def test_map_rows_respects_limit_and_rejects_unknown_dataset():
    rows = [{"question": f"q{i}", "answer": f"#### {i}"} for i in range(10)]
    assert len(map_rows("gsm8k", rows, limit=3)) == 3

    try:
        map_rows("inexistente", rows, limit=1)
    except ValueError as e:
        assert "inexistente" in str(e)
    else:
        raise AssertionError("debió lanzar ValueError")
