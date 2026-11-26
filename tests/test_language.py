from adapters.fake_llm import ScriptedLLM
from adapters.fake_search import StaticSearch
from adapters.in_memory_kb import InMemoryKB
from application.investigator import Investigator
from application.language import UNKNOWN, question_language
from application.naive_rag import naive_rag
from application.redactor import Redactor
from domain.evidence import Evidence
from domain.report import Report


def test_question_language_detects_english_and_spanish():
    assert question_language("how many xbox 360 games are there") == "inglés"
    assert question_language("what bird family is the owl") == "inglés"
    assert question_language("¿Cuál es la capital del Perú?") == "español"
    assert question_language("Strigidae") == UNKNOWN


def test_question_language_only_reads_the_statement_of_mbpp_questions():
    mbpp = ("Write a function to find the shared elements from the given two lists.\n"
            "Tu código debe pasar estas pruebas:\nassert similar_elements((3, 4), (4, 5)) == (4,)")
    assert question_language(mbpp) == "inglés"


def test_agents_and_baseline_ask_for_the_language_of_the_question():
    question = "what bird family is the owl"
    llm = ScriptedLLM(["Final: Strigidae", "The owl belongs to the family Strigidae.", "Strigidae"])
    Investigator(llm, StaticSearch([]), InMemoryKB([])).run(question)
    Redactor(llm, task="qa").write(Report(question, "La familia es Strigidae."))
    naive_rag(question, llm, InMemoryKB([Evidence("wiki", "Owls are in the family Strigidae.")]))
    assert all("inglés" in prompt for prompt in llm.prompts)
