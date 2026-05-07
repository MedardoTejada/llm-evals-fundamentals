"""
Capa de abstracción del proveedor LLM.

Este es el único archivo que necesitas modificar para cambiar de proveedor.
Actualmente configurado para: Groq (llama-3.3-70b-versatile).

Exporta tres funciones usadas en todo el proyecto:
  - get_judge()             → juez para deepeval (evalúa los outputs)
  - completar()             → generación de texto simple (RAG, Fase 1)
  - completar_con_tools()   → llamada con function calling (agente)
"""
import os
from groq import Groq
from dotenv import load_dotenv
from deepeval.models.base_model import DeepEvalBaseLLM

load_dotenv()

# ── Configuración central ─────────────────────────────────────────────────────

_API_KEY = os.environ["GROQ_API_KEY"]
_MODELO_ID = "llama-3.3-70b-versatile"

_client = Groq(api_key=_API_KEY)


# ── Juez para deepeval ────────────────────────────────────────────────────────

class GroqJudge(DeepEvalBaseLLM):
    """
    Wrapper que conecta Groq a deepeval como modelo juez.
    Groq no tiene soporte nativo en deepeval, así que usamos el patrón
    DeepEvalBaseLLM — el mismo que usaríamos para cualquier proveedor custom.
    """
    def __init__(self):
        self.client = _client

    def load_model(self):
        return self.client

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=_MODELO_ID,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4096,
        )
        return response.choices[0].message.content

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return _MODELO_ID


# ── Interfaz pública ──────────────────────────────────────────────────────────

def get_judge() -> GroqJudge:
    """Devuelve el modelo juez que deepeval usa para puntuar outputs."""
    return GroqJudge()


def completar(prompt: str, instruccion_sistema: str | None = None) -> str:
    """Genera texto a partir de un prompt. Opcionalmente acepta una instrucción de sistema."""
    messages = []
    if instruccion_sistema:
        messages.append({"role": "system", "content": instruccion_sistema})
    messages.append({"role": "user", "content": prompt})

    response = _client.chat.completions.create(
        model=_MODELO_ID,
        messages=messages,
        max_tokens=512,
    )
    return response.choices[0].message.content


def completar_con_tools(messages: list, tools: list) -> tuple[str | None, list, dict]:
    """
    Hace una llamada al modelo con herramientas disponibles.
    Devuelve (contenido, tool_calls, mensaje_asistente_dict).
    El mensaje_asistente_dict está listo para agregarse al historial de mensajes.
    """
    response = _client.chat.completions.create(
        model=_MODELO_ID,
        messages=messages,
        tools=tools,
    )
    msg = response.choices[0].message

    asst_dict = {"role": "assistant", "content": msg.content}
    if msg.tool_calls:
        asst_dict["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in msg.tool_calls
        ]

    return msg.content, msg.tool_calls or [], asst_dict
