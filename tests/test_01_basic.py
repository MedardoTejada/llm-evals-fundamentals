"""
Fase 1 — Fundamentos de evaluación single-turn.

Conceptos cubiertos:
  - LLMTestCase: el primitivo central de deepeval
  - GEval: métrica personalizada LLM-as-a-judge con tus propios criterios
  - AnswerRelevancyMetric: ¿la respuesta aborda la pregunta?
  - assert_test: hook de integración con pytest
  - SingleTurnParams: parámetros que el juez puede ver al evaluar (renombrado en v3.7)

Ejecutar con:
    deepeval test run tests/test_01_basic.py
"""
import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.metrics import GEval, AnswerRelevancyMetric
from src.llm_client import get_judge, completar

judge = get_judge()


def preguntar_llm(pregunta: str) -> str:
    """Sistema bajo prueba: una llamada directa al LLM sin RAG ni herramientas."""
    return completar(pregunta)


# ── Definición de métricas ────────────────────────────────────────────────────

metrica_claridad = GEval(
    name="Claridad",
    criteria=(
        "La respuesta es clara, concisa y fácil de entender. "
        "Evita jerga innecesaria y va directo al punto."
    ),
    evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
    threshold=0.7,
    model=judge,
)

metrica_precision = GEval(
    name="Precisión",
    criteria=(
        "La respuesta es factualmente precisa y no contiene alucinaciones. "
        "Las afirmaciones son consistentes con la salida esperada."
    ),
    evaluation_params=[
        SingleTurnParams.INPUT,
        SingleTurnParams.ACTUAL_OUTPUT,
        SingleTurnParams.EXPECTED_OUTPUT,
    ],
    threshold=0.7,
    model=judge,
)

metrica_relevancia = AnswerRelevancyMetric(threshold=0.7, model=judge)


# ── Casos de prueba ───────────────────────────────────────────────────────────

casos_de_prueba = [
    {
        "input": "¿Cuál es la capital de Francia?",
        "expected_output": "La capital de Francia es París.",
    },
    {
        "input": "Explica qué es un modelo de lenguaje grande en una sola oración.",
        "expected_output": (
            "Un modelo de lenguaje grande es una red neuronal entrenada con enormes "
            "cantidades de texto que puede generar y comprender lenguaje humano."
        ),
    },
    {
        "input": "¿Cuánto es el 15% de 200?",
        "expected_output": "El 15% de 200 es 30.",
    },
]


@pytest.mark.parametrize("caso", casos_de_prueba)
def test_llm_claridad_y_relevancia(caso):
    """Cada respuesta debe ser clara y relevante para la pregunta formulada."""
    salida_actual = preguntar_llm(caso["input"])
    test_case = LLMTestCase(
        input=caso["input"],
        actual_output=salida_actual,
        expected_output=caso["expected_output"],
    )
    assert_test(test_case, [metrica_claridad, metrica_relevancia])


@pytest.mark.parametrize("caso", casos_de_prueba)
def test_llm_precision(caso):
    """Cada respuesta debe ser factualmente precisa respecto a la salida esperada."""
    salida_actual = preguntar_llm(caso["input"])
    test_case = LLMTestCase(
        input=caso["input"],
        actual_output=salida_actual,
        expected_output=caso["expected_output"],
    )
    assert_test(test_case, [metrica_precision])
