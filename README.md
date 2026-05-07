# llm-evals-fundamentals

Un proyecto para aprender a evaluar aplicaciones LLM desde cero usando [deepeval](https://deepeval.com/).

Si nunca has escrito un eval, este es tu punto de partida. Cada fase agrega un concepto nuevo, con código comentado que explica el por qué de cada decisión.

---

## ¿Qué es un eval?

Un eval (evaluación) es un test que mide la **calidad** de la respuesta de un LLM, no solo si corrió sin errores.

A diferencia de un test tradicional (verdadero/falso), los evals devuelven una **puntuación entre 0 y 1** junto con un razonamiento. Quien puntúa es otro LLM que actúa como juez.

```
Pregunta → LLM → Respuesta → Juez → Score (0.0 a 1.0) + Razón
```

---

## Fases del proyecto

| Fase | Concepto | Estado |
|------|----------|--------|
| **01 · Básico** | LLMTestCase, GEval, AnswerRelevancy | ✅ Listo |
| **02 · RAG** | retrieval_context, Faithfulness, ContextualRelevancy | 🔜 Próximamente |
| **03 · Agente** | tools_called, TaskCompletion, ToolCorrectness | 🔜 Próximamente |

---

## Fase 1 — Lo esencial

### El primitivo central: `LLMTestCase`

Todo eval en deepeval empieza con un `LLMTestCase`. Es el objeto que agrupa lo que necesita el juez para evaluar:

```python
from deepeval.test_case import LLMTestCase

test_case = LLMTestCase(
    input="¿Cuál es la capital de Francia?",   # la pregunta que le hiciste al LLM
    actual_output="La capital de Francia es París.",  # lo que respondió
    expected_output="París.",                  # lo que esperabas (opcional)
)
```

### Las métricas

**GEval** — defines tú el criterio en lenguaje natural:

```python
from deepeval.metrics import GEval
from deepeval.test_case import SingleTurnParams

metrica = GEval(
    name="Claridad",
    criteria="La respuesta es clara y concisa, sin información innecesaria.",
    evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
    threshold=0.7,  # mínimo para pasar
    model=judge,
)
```

**AnswerRelevancyMetric** — integrada en deepeval, mide si la respuesta contesta la pregunta:

```python
from deepeval.metrics import AnswerRelevancyMetric

metrica = AnswerRelevancyMetric(threshold=0.7, model=judge)
```

### El juez

deepeval necesita un LLM para evaluar. Como usamos Groq (gratuito), creamos un wrapper con `DeepEvalBaseLLM`:

```python
# src/llm_client.py
class GroqJudge(DeepEvalBaseLLM):
    def generate(self, prompt: str) -> str:
        # le decimos a deepeval cómo llamar a Groq
        ...
```

Este patrón funciona para cualquier proveedor que no tenga soporte nativo en deepeval.

---

## Cómo correrlo

### 1. Requisitos
- Python 3.11+
- Cuenta en [console.groq.com](https://console.groq.com) (gratuito)

### 2. Instalar
```bash
git clone https://github.com/MedardoTejada/llm-evals-fundamentals.git
cd llm-evals-fundamentals

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configurar API key
```bash
cp .env.example .env
# edita .env y agrega tu GROQ_API_KEY
```

### 4. Correr los evals
```bash
deepeval test run tests/test_01_basic.py
```

Verás una tabla con el score de cada caso, si pasó o falló, y el razonamiento del juez.

---

## Resultado esperado

Al correr la Fase 1 verás algo así:

```
✓ Evaluation completed!
» Test Results (6 total tests):
   » Pass Rate: 83.3% | Passed: 5 | Failed: 1
```

El fallo es intencional: el LLM respondió con más información de la necesaria y el juez lo penalizó por falta de concisión. Así funciona un eval — no todo tiene que pasar.

---

## Stack

- [deepeval](https://deepeval.com/) — framework de evaluación
- [Groq](https://groq.com/) — LLM gratuito (llama-3.3-70b-versatile)
- [python-dotenv](https://github.com/theskumar/python-dotenv) — variables de entorno
