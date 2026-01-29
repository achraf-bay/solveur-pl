import json
import re
import google.generativeai as genai


class LLMExtractor:

    def __init__(self, api_key=""):
        self.api_key = api_key

    def set_api_key(self, api_key):
        self.api_key = api_key

    def extract_problem(self, text):

        if not self.api_key:
            raise Exception("API Key non configurée.")

        # CONFIG
        genai.configure(api_key=self.api_key)

        # LISTE DE MODÈLES 
        model_names = [
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
            "gemini-2.0-pro",
            "gemini-1.5-flash",
            "gemini-1.5-pro",
            "gemini-2.5-flash"
        ]


        model = None
        last_error = None

        # Sélection d’un modèle valide
        for model_name in model_names:
            try:
                test_model = genai.GenerativeModel(model_name)
                _ = test_model.generate_content("test")
                print(f"✅ Modèle OK: {model_name}")
                model = test_model
                break
            except Exception as e:
                last_error = str(e)

        if model is None:
            raise Exception(f"Aucun modèle Gemini valide. Erreur: {last_error}")

        # Prompt d'extraction
        prompt = self._build_extraction_prompt(text)

        # Appel API
        response = model.generate_content(prompt)
        result_text = response.text.strip()

        # parsing JSON
        result = self._parse_response(result_text)

        # Validation
        self._validate_result(result)

        return result

    def _build_extraction_prompt(self, text):
        return f"""
Analyse le problème et renvoie uniquement un JSON strict.

Problème:
{text}

Format JSON à respecter:
{{
  "objective_type": "max" ou "min",
  "c": [coef_x1, coef_x2],
  "constraints": [
    {{"a": nombre, "b": nombre, "op": "<=", "c": nombre}}
  ]
}}

Réponds uniquement en JSON.
"""

    def _parse_response(self, result_text):
        if "```" in result_text:
            parts = result_text.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("{") and part.endswith("}"):
                    result_text = part
                    break

        return json.loads(result_text)

    def _validate_result(self, result):

        if "objective_type" not in result:
            raise Exception("objective_type manquant")

        if result["objective_type"] not in ("max", "min"):
            raise Exception("objective_type invalide")

        if "c" not in result or len(result["c"]) != 2:
            raise Exception("'c' doit contenir 2 coefficients")

        if "constraints" not in result or not isinstance(result["constraints"], list):
            raise Exception("'constraints' doit être une liste")

        if len(result["constraints"]) == 0:
            raise Exception("Aucune contrainte détectée")

        for cons in result["constraints"]:
            for key in ("a", "b", "op", "c"):
                if key not in cons:
                    raise Exception(f"Contrainte invalide, clé manquante: {key}")

            if cons["op"] not in ["<=", ">=", "="]:
                raise Exception(f"Opérateur invalide: {cons['op']}")


# Exemple d'utilisation
if __name__ == "__main__":
    API_KEY = "TON_API_KEY"

    text_exemple = """
    Une entreprise dispose de 10 000 m²...
    """

    extractor = LLMExtractor(API_KEY)

    try:
        result = extractor.extract_problem(text_exemple)
        print(json.dumps(result, indent=2, ensure_ascii=False))

    except Exception as e:
        print(f"❌ Erreur: {e}")

