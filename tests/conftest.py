"""
Configuración global de pytest para el proyecto.

Agrega una pausa entre cada test para respetar el rate limit del tier gratuito
de Gemini (15 req/min). En producción con una API key de pago esto no es necesario.
"""
import time
import pytest


@pytest.fixture(autouse=True)
def pausar_entre_tests():
    yield
    time.sleep(8)
