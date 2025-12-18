# api/test_endpoints.py

import pytest
from fastapi.testclient import TestClient
from api.endpoints import _contains_sensitive_content
from main import app

@pytest.mark.parametrize("text,expected", [
    ("Mi contraseña es secreta", True),
    ("El password debe ser seguro", True),
    ("Número de tarjeta: 1234", True),
    ("Esta es una credencial importante", True),
    ("El secreto está guardado", True),
    ("No hay datos sensibles aquí", False),
    ("Este texto es completamente inocuo", False),
    ("Información pública", False),
])
def test_contains_sensitive_content(text, expected):
    assert _contains_sensitive_content(text) == expected

@pytest.mark.parametrize("question", [
    "¿Cuál es el objetivo principal de la tesis doctoral de Alejandra E. Suárez?",
    "¿Qué objetos de estudio específicos se incluyen en esta investigación?",
    "¿Qué instrumentos espaciales de rayos X proporcionaron los datos principales para este estudio?",
    "¿Qué hallazgo se destaca respecto al fragmento G del remanente de Vela?",
    "¿Cuál es el tipo de progenitor propuesto para el remanente G306.3-0.9?",
    "¿Qué caracteriza a los remanentes de 'Morfología Mixta' (MM)?",
    "¿Qué fenómeno físico se propone para explicar la presencia de fragmentos ricos en Si en el interior de Vela?",
    "¿Cuál es la morfología peculiar del remanente G332.5-5.6 en radio?",
    "¿Cuántas supernovas por siglo menciona la tesis que ocurren en nuestra galaxia?",
    "¿En qué año se defendió esta tesis según la portada?",
    "¿Qué capítulo de la tesis describe el descubrimiento de vida bacteriana en el fragmento G de Vela?",
    "¿Cuál es el nombre del telescopio terrestre en la Antártida que se usó para validar los datos?",
    "Según la tesis, ¿qué influencia tuvo la Luna en la observación de G309.2-0.6?",
    "What are the three main categories of supernova remnants according to the classification in Chapter 2?",
])
def test_rag_ask_endpoint(question):
    client = TestClient(app)
    response = client.post("/ask", json={"question": question})
    assert response.status_code == 200
    data = response.json()
    assert "question" in data
    assert "answer" in data
    assert "grounded" in data
    # Optionally print or further check the answer
    print(f"Q: {question}\nA: {data['answer']}\n")