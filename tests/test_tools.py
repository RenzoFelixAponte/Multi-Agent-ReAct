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


MBPP_QUESTION = "Suma dos números.\nTu código debe pasar estas pruebas:\nassert add(1, 2) == 3\nassert add(0, 0) == 0"


def test_extract_asserts_reads_the_tests_from_the_question():
    from application.answer_checks import extract_asserts

    assert extract_asserts(MBPP_QUESTION) == ["assert add(1, 2) == 3", "assert add(0, 0) == 0"]
    assert extract_asserts("¿Capital del Perú?") == []


def test_auditor_runs_the_delivered_code_against_the_question_tests():
    from application.auditor import Auditor
    from domain.report import Report

    auditor = Auditor(ScriptedLLM([]), runner=SubprocessCodeRunner())  # no debe llamar al LLM
    good = "```python\ndef add(a, b):\n    return a + b\n```"
    assert auditor.review(Report(MBPP_QUESTION, good, ())).approved

    wrong = auditor.review(Report(MBPP_QUESTION, "```python\ndef add(a, b):\n    return a - b\n```", ()))
    assert not wrong.approved and "no pasa las pruebas" in wrong.reason and "AssertionError" in wrong.reason

    no_code = auditor.review(Report(MBPP_QUESTION, "La función funciona.", ()))
    assert not no_code.approved and "bloque" in no_code.reason


def test_investigator_delivers_the_executed_code_when_final_omits_it():
    code = "Action: python[def add(a, b):\n    return a + b\n\nassert add(1, 2) == 3\nprint('ok')]"
    llm = ScriptedLLM([code, "Final: La función pasó las pruebas."])
    report = Investigator(llm, StaticSearch([]), InMemoryKB([]), runner=SubprocessCodeRunner(),
                          tools=("python",), task="code").run(MBPP_QUESTION)
    assert report.answer.startswith("La función pasó las pruebas.")
    assert "```python\ndef add(a, b):" in report.answer


def test_investigator_only_offers_the_tools_of_its_task():
    llm = ScriptedLLM(["Action: kb[huevos]", "Action: calc[16-3-4]", "Final: Respuesta final: 9"])
    kb = InMemoryKB([Evidence("wiki", "Janet tiene 16 huevos.")])
    report = Investigator(llm, StaticSearch([]), kb, tools=("calc",), task="math").run("huevos")

    assert "kb[" not in llm.prompts[0].split("Ejemplo:")[0]
    assert "calc[50 - (4*3 + 2*5)]" in llm.prompts[0]  # ejemplo resuelto de matemáticas
    assert "herramienta kb no disponible" in llm.prompts[1]
    assert [e.source for e in report.evidence] == ["calc"]


def test_investigator_drops_duplicated_evidence():
    llm = ScriptedLLM(["Action: kb[capital Perú]", "Final: Lima"])
    kb = InMemoryKB([Evidence("wiki", "Lima es la capital del Perú.")])
    report = Investigator(llm, StaticSearch([]), kb).run("capital", prefetched=kb.retrieve("capital Perú"))
    assert len(report.evidence) == 1


def test_code_flow_audits_by_execution_and_retries_with_the_error_and_previous_answer():
    llm = ScriptedLLM([
        "Final:\n```python\ndef add(a, b):\n    return a - b\n```",           # ronda 1: código mal
        "Action: python[def add(a, b):\n    return a + b\n\nassert add(1, 2) == 3]",
        "Final: Corregido.",                                                  # ronda 2: sin bloque
    ])
    runner = SubprocessCodeRunner()
    flow = build_flow(llm, StaticSearch([]), InMemoryKB([]), runner=runner, task="code")
    state = flow.invoke({"question": MBPP_QUESTION})

    assert state["verdict"].approved and state["rounds"] == 2
    assert "AssertionError" in llm.prompts[1] and "return a - b" in llm.prompts[1]
    item = BenchmarkItem("mbpp", "1", "suma", "", tests=("assert add(1, 2) == 3", "assert add(0, 0) == 0"))
    assert passes_tests(state["final"], item, runner)


def test_auditor_execution_becomes_evidence_of_the_approved_answer():
    # El Investigador escribe el código directo en Final, sin ejecutarlo: lo único que lo respalda
    # es la ejecución del Auditor, que debe quedar como evidencia para el juez.
    llm = ScriptedLLM(["Final:\n```python\ndef add(a, b):\n    return a + b\n```"])
    flow = build_flow(llm, StaticSearch([]), InMemoryKB([]), runner=SubprocessCodeRunner(), task="code")
    state = flow.invoke({"question": MBPP_QUESTION})

    assert state["verdict"].approved and state["rejections"] == []
    assert [e.source for e in state["report"].evidence] == ["auditor"]
    assert "assert add(1, 2) == 3" in state["report"].evidence[0].text


GSM_QUESTION = "Janet has 16 eggs. She eats 3 and bakes with 4. She sells the rest at $2 each. How much does she make?"
CALC = (Evidence("calc", "(16-3-4)*2 = 18"),)


def test_math_auditor_solves_on_its_own_and_approves_when_the_numbers_match():
    from application.auditor import Auditor
    from domain.report import Report

    llm = ScriptedLLM(["16 - 3 - 4 = 9 eggs, 9 * 2 = 18.\nRespuesta final: 18"])
    verdict = Auditor(llm, task="math").review(Report(GSM_QUESTION, "She makes $18.\nRespuesta final: $18", CALC))
    assert verdict.approved
    assert GSM_QUESTION in llm.prompts[0] and "$18" not in llm.prompts[0]  # no ve la respuesta del Investigador
    assert verdict.evidence == ()  # su resolución no cuenta como evidencia


def test_math_auditor_rejects_a_mismatch_with_both_numbers_and_solves_only_once():
    from application.auditor import Auditor
    from domain.report import Report

    llm = ScriptedLLM(["Respuesta final: 18"])
    auditor = Auditor(llm, task="math")
    wrong = auditor.review(Report(GSM_QUESTION, "Respuesta final: 32", CALC))
    assert not wrong.approved and "18" in wrong.reason and "32" in wrong.reason
    assert auditor.review(Report(GSM_QUESTION, "Respuesta final: 18", CALC)).approved
    assert len(llm.prompts) == 1  # la segunda ronda reutiliza su resolución


def test_math_auditor_falls_back_to_the_llm_review_when_it_cannot_solve():
    from application.auditor import Auditor
    from domain.report import Report

    llm = ScriptedLLM(["no sé resolverlo", "APROBADO"])
    assert Auditor(llm, task="math").review(Report(GSM_QUESTION, "Respuesta final: 18", CALC)).approved
    assert "Verifica si la respuesta" in llm.prompts[1]

    no_number = Auditor(ScriptedLLM([]), task="math").review(Report(GSM_QUESTION, "Gana dinero.", CALC))
    assert not no_number.approved and "Respuesta final" in no_number.reason


def test_math_flow_retries_with_the_independent_result_and_approves_the_fix():
    llm = ScriptedLLM([
        "Thought: 16 - 3 eggs at $2.\nAction: calc[(16-3)*2]", "Final: She makes $26.\nRespuesta final: 26",
        "9 eggs at $2.\nRespuesta final: 18",                                   # Auditor, ronda 1
        "Thought: I forgot the 4 eggs.\nAction: calc[(16-3-4)*2]", "Final: She makes $18.\nRespuesta final: 18",
        "She makes 18 dollars.\nRespuesta final: 18",                           # Redactor
    ])
    flow = build_flow(llm, StaticSearch([]), InMemoryKB([]), task="math")
    state = flow.invoke({"question": GSM_QUESTION})
    assert state["verdict"].approved and state["rounds"] == 2
    assert "da 18 y tu respuesta da 26" in state["rejections"][0]
    assert "da 18 y tu respuesta da 26" in llm.prompts[3]  # el motivo llega al Investigador
