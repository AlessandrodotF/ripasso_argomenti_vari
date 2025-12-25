from pydantic import BaseModel
from ollama import chat


# semplice domanda su una nazione
class Country(BaseModel):
    name: str
    capital: str
    languages: str


class Object(BaseModel):
    name: str
    confidence: float
    attrbutes: str


class ImageDescription(
    BaseModel
):  # da continuare https://docs.ollama.com/capabilities/structured-outputs#python-2
    summary: str


response = chat(
    model="llama3.2:1b",
    messages=[{"role": "user", "content": "tell me about Italy"}],
    format=Country.model_json_schema(),
    options={"temperature": 0},
)
print(response.message.content)
