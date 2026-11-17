import pytest

from adapters.fake_llm import ScriptedLLM
from adapters.fake_search import StaticSearch
from adapters.in_memory_kb import InMemoryKB
from adapters.subprocess_code_runner import SubprocessCodeRunner
from application.answer_checks import extract_code, extract_final_number, numbers_match, passes_tests
from application.calculator import calculate
from application.graph import build_flow
from application.investigator import Investigator, parse_action
from domain.benchmark_item import BenchmarkItem
from domain.evidence import Evidence


def test_calculate_handles_arithmetic_and_formats_results():
    assert calculate("(16 - 3 - 4) * 2") == "18"
    assert calculate("10 / 4") == "2.5"
    assert calculate("2 ^ 5") == "32"
    assert calculate("round(7 / 3, 2)") == "2.33"
    assert calculate("16-3-4 = 9") == "9"  # ignora un '= resultado' escrito por el modelo


@pytest.mark.parametrize("expr", ["__import__('os').system('ls')", "open('x')", "1 +", "", "1 / 0", "9 ** 999"])
def test_calculate_rejects_anything_that_is_not_plain_arithmetic(expr):
    with pytest.raises(ValueError):
        calculate(expr)


def test_parse_action_keeps_nested_brackets_and_stops_at_next_section():
    assert parse_action("Thought: x\nAction: kb[capital Perú]") == ("kb", "capital Perú")
    assert parse_action("Action: python[print(sum([1, 2]))]\nFinal: no") == ("python", "print(sum([1, 2]))")
    assert parse_action("Action: calc[2+2]\nAction: web[otra]") == ("calc", "2+2")
    assert parse_action("Final: 4") is None


def test_final_number_prefers_final_line_and_handles_thousands():
    assert extract_final_number("Gana 16 y vende 3.\nRespuesta final: $1,200") == 1200
    assert extract_final_number("el total es 18 dólares") == 18
    assert extract_final_number("sin números") is None
    assert numbers_match("Respuesta final: 18.0", "18")
    assert not numbers_match("Respuesta final: 19", "18")


def test_extract_code_takes_fenced_block_or_whole_text():
    assert extract_code("texto\n```python\ndef f():\n    return 1\n```\nfin") == "def f():\n    return 1"
    assert extract_code("def f(): return 1") == "def f(): return 1"


def test_runner_returns_output_errors_and_timeouts():
    runner = SubprocessCodeRunner()
    assert runner.run("print(2 + 2)").output == "4"
    failed = runner.run("raise ValueError('boom')")
    assert not failed.ok and "boom" in failed.output
    slow = runner.run("while True: pass", timeout=1)
    assert not slow.ok and "tiempo límite" in slow.output


def test_runner_limits_memory():
    result = SubprocessCodeRunner(memory_mb=256).run("x = bytearray(2 * 1024 ** 3)", timeout=5)
    assert not result.ok


def test_passes_tests_runs_asserts_from_mbpp():
    item = BenchmarkItem("mbpp", "1", "suma", "", tests=("assert add(1, 2) == 3",))
    runner = SubprocessCodeRunner()
    assert passes_tests("```python\ndef add(a, b):\n    return a + b\n```", item, runner)
    assert not passes_tests("def add(a, b):\n    return a - b", item, runner)
    assert not passes_tests("def add(a, b): return a + b", BenchmarkItem("mbpp", "2", "q", ""), runner)


def test_investigator_uses_calc_and_records_it_as_evidence():
    llm = ScriptedLLM(["Action: calc[(16-3-4)*2]", "Final: Respuesta final: 18"])
    report = Investigator(llm, StaticSearch([]), InMemoryKB([])).run("¿cuánto gana?")

    assert report.answer == "Respuesta final: 18"
    assert [(e.source, e.text) for e in report.evidence] == [("calc", "(16-3-4)*2 = 18")]
    assert "Observation: 18" in llm.prompts[1]


def test_investigator_reports_calc_errors_without_evidence():
    llm = ScriptedLLM(["Action: calc[1/0]", "Final: no se pudo"])
    report = Investigator(llm, StaticSearch([]), InMemoryKB([])).run("q")
    assert report.evidence == ()
    assert "error de cálculo: división entre cero" in llm.prompts[1]


def test_investigator_runs_python_only_when_a_runner_is_given():
    code = "Action: python[def add(a, b): return a + b\nprint(add(1, 2))]"
    with_runner = Investigator(ScriptedLLM([code, "Final: ok"]), StaticSearch([]), InMemoryKB([]),
                               runner=SubprocessCodeRunner())
    report = with_runner.run("suma")
    assert report.evidence[0].source == "python"
    assert report.evidence[0].text.endswith("salida: 3")

    llm = ScriptedLLM([code, "Final: ok"])
    Investigator(llm, StaticSearch([]), InMemoryKB([])).run("suma")
    assert "herramienta python no disponible" in llm.prompts[1]
    assert "python[código]" not in llm.prompts[0]


def test_flow_solves_a_math_question_with_calc():
    llm = ScriptedLLM([
        "Action: calc[16-3-4]", "Final: Respuesta final: 9",   # investigador
        "APROBADO",                                             # auditor
        "Tiene 9.\nRespuesta final: 9",                         # redactor
    ])
    flow = build_flow(llm, StaticSearch([]), InMemoryKB([]))
    state = flow.invoke({"question": "Janet tiene 16 huevos..."})
    assert state["verdict"].approved
    assert numbers_match(state["final"], "9")


def test_investigator_blocks_a_repeated_action_instead_of_looping():
    llm = ScriptedLLM(["Action: kb[capital Perú]", "Action: kb[capital Perú]", "Final: Lima"])
    kb = InMemoryKB([Evidence("wiki", "Lima es la capital del Perú.")])
    report = Investigator(llm, StaticSearch([]), kb).run("capital Perú")

    assert report.answer == "Lima"
    assert len(report.evidence) == 1  # la repetición no volvió a ejecutar la herramienta
    assert "No la repitas" in llm.prompts[2]


def test_last_step_asks_for_a_final_answer():
    llm = ScriptedLLM(["Action: calc[1+1]", "Final: 2"])
    Investigator(llm, StaticSearch([]), InMemoryKB([]), max_steps=2).run("q")
    assert "último paso" not in llm.prompts[0]
    assert "último paso" in llm.prompts[1]


def test_redactor_returns_code_untouched_instead_of_rewriting_it():
    from application.redactor import Redactor
    from domain.report import Report

    llm = ScriptedLLM([])  # si el redactor llamara al LLM, ScriptedLLM fallaría
    answer = "Final:\n```python\ndef add(a, b):\n    return a + b\n```"
    assert Redactor(llm).write(Report("suma", answer, ())) == answer


def test_investigator_warns_when_the_model_writes_several_actions():
    llm = ScriptedLLM(["Action: calc[1+1]\nAction: calc[2+2]", "Final: 2"])
    report = Investigator(llm, StaticSearch([]), InMemoryKB([])).run("q")
    assert [e.text for e in report.evidence] == ["1+1 = 2"]
    assert "Solo se ejecutó la primera Action" in llm.prompts[1]



def test_auditor_rejects_code_that_was_never_executed_as_delivered():
    from application.auditor import Auditor
    from domain.report import Report

    answer = "```python\ndef add(a, b):\n    return a + b\nassert add(1, 2) == 3\n```"
    ran_only_the_def = (Evidence("python", "código:\ndef add(a, b):\n    return a + b\nsalida: (sin salida)"),)
    verdict = Auditor(ScriptedLLM([])).review(Report("suma", answer, ran_only_the_def))
    assert not verdict.approved and "no se ejecutó tal cual" in verdict.reason

    ran_everything = (Evidence("python", "código:\ndef add(a, b):\n  return a + b\n\nassert add(1, 2) == 3\nsalida: ok"),)
    assert Auditor(ScriptedLLM(["APROBADO"])).review(Report("suma", answer, ran_everything)).approved
