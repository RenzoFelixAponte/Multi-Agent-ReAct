from adapters.hf_mappers import map_mbpp, map_rows, wikiqa_documents
from domain.evidence import Evidence


def test_wikiqa_groups_by_question_and_keeps_distractors_in_context():
    rows = [
        {"question_id": "Q1", "question": "capital del Perú?", "document_title": "Perú",
         "answer": "Lima es la capital.", "label": "1"},
        {"question_id": "Q1", "question": "capital del Perú?", "document_title": "Perú",
         "answer": "El Perú tiene muchos ríos.", "label": "0"},
        {"question_id": "Q2", "question": "sin respuesta correcta?", "document_title": "X",
         "answer": "irrelevante", "label": "0"},
    ]
    items = map_rows("wiki_qa", rows, limit=10)
    assert len(items) == 1  # Q2 no tiene frase correcta
    assert items[0].item_id == "Q1"
    assert items[0].reference == "Lima es la capital."
    assert items[0].context == (
        Evidence("Perú", "Lima es la capital."),
        Evidence("Perú", "El Perú tiene muchos ríos."),
    )


def test_wikiqa_reference_joins_all_positive_sentences():
    rows = [
        {"question_id": "Q1", "question": "q", "document_title": "T", "answer": "a1", "label": 1},
        {"question_id": "Q1", "question": "q", "document_title": "T", "answer": "a2", "label": 1},
    ]
    assert map_rows("wiki_qa", rows, limit=5)[0].reference == "a1 a2"


def test_wikiqa_documents_include_distractors_without_duplicates():
    rows = [
        {"question_id": "Q1", "question": "q1", "document_title": "T", "answer": "bien", "label": 1},
        {"question_id": "Q1", "question": "q1", "document_title": "T", "answer": "distractor", "label": 0},
        {"question_id": "Q2", "question": "q2", "document_title": "T", "answer": "bien", "label": 1},
        {"question_id": "Q2", "question": "q2", "document_title": "T", "answer": "distractor", "label": 0},
    ]
    docs = wikiqa_documents(map_rows("wiki_qa", rows, limit=5))
    assert docs == [Evidence("T", "bien"), Evidence("T", "distractor")]


def test_gsm8k_extracts_final_number_and_drops_rows_without_marker():
    rows = [
        {"question": "Janet tiene 16 huevos...", "answer": "16 - 3 - 4 = <<16-3-4=9>>9\n#### 18"},
        {"question": "sin marca", "answer": "solo texto"},
        {"question": "miles", "answer": "razonamiento\n#### 1,200"},
    ]
    items = map_rows("gsm8k", rows, limit=10)
    assert [i.reference for i in items] == ["18", "1200"]


def test_mbpp_uses_text_as_question_and_code_as_reference():
    row = {"task_id": 11, "text": "Remove first occurrence", "code": "def f(s): return s",
           "test_list": ["assert f('a') == 'a'", "assert f('') == ''"], "test_setup_code": ""}
    item = map_mbpp(row, 0)
    assert item.item_id == "11"
    assert item.question == "Remove first occurrence"
    assert item.reference == "def f(s): return s"
    assert item.tests == ("assert f('a') == 'a'", "assert f('') == ''")


def test_map_rows_respects_limit_and_rejects_unknown_dataset():
    rows = [{"question": f"q{i}", "answer": f"#### {i}"} for i in range(10)]
    assert len(map_rows("gsm8k", rows, limit=3)) == 3

    try:
        map_rows("inexistente", rows, limit=1)
    except ValueError as e:
        assert "inexistente" in str(e)
    else:
        raise AssertionError("debió lanzar ValueError")
