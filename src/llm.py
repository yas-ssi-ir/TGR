"""
Wrapper du LLM local via l'API HTTP d'Ollama.
Supporte le mode complet (generate) et le streaming token par token.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import LLM_MAX_TOKENS, LLM_MODEL, LLM_TEMPERATURE, OLLAMA_URL

import requests


class OllamaLLM:
    def __init__(self, model: str = LLM_MODEL, base_url: str = OLLAMA_URL):
        self.model = model
        self.base_url = base_url

    def is_available(self) -> bool:
        """Vérifie qu'Ollama tourne et que le modèle est présent."""
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=3)
            models = [m["name"] for m in r.json().get("models", [])]
            return any(self.model in m for m in models)
        except Exception:
            return False

    def generate(self, system: str, user: str,
                 temperature: float = LLM_TEMPERATURE,
                 max_tokens: int = LLM_MAX_TOKENS) -> str:
        """Génération complète (bloquante). Utilisée par les nœuds de décision."""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "keep_alive": "30m",   # garde le modèle chargé en RAM entre les appels
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        r = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=300)
        r.raise_for_status()
        return r.json()["message"]["content"].strip()

    def generate_stream(self, system: str, user: str,
                        temperature: float = LLM_TEMPERATURE,
                        max_tokens: int = LLM_MAX_TOKENS):
        """Génération en streaming — yield chaque fragment de texte.
        Indispensable sur CPU : l'usager voit la réponse s'écrire."""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": True,
            "keep_alive": "30m",   # garde le modèle chargé en RAM entre les appels
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        with requests.post(f"{self.base_url}/api/chat", json=payload,
                           stream=True, timeout=300) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                data = json.loads(line)
                chunk = data.get("message", {}).get("content", "")
                if chunk:
                    yield chunk
                if data.get("done"):
                    break

    def decide(self, system: str, user: str) -> str:
        """Décision rapide OUI/NON — température 0, très peu de tokens."""
        return self.generate(system, user, temperature=0.0, max_tokens=8).upper()


if __name__ == "__main__":
    llm = OllamaLLM()
    if not llm.is_available():
        print(f"Ollama indisponible ou modèle '{llm.model}' absent.")
        print(f"→ Installez Ollama puis : ollama pull {llm.model}")
        sys.exit(1)
    print(llm.generate("Tu es un assistant utile.", "Dis bonjour en une phrase en français."))
