import google.generativeai as genai

# ⭐ METTEZ VOTRE CLE ICI
API_KEY = "AIzaSyC47OsbbkdzmOnpduoV3UplTJIrzGtEQZU"

genai.configure(api_key=API_KEY)

print("=" * 60)
print("🔍 DIAGNOSTIC COMPLET GEMINI")
print("=" * 60)

# Version de la bibliothèque
import google.generativeai as genai_module
print(f"\n📦 Version google-generativeai: {genai_module.__version__}")

# Liste des modèles
print("\n📋 Modèles disponibles:\n")

try:
    models = list(genai.list_models())
    
    if not models:
        print("❌ Aucun modèle trouvé - Problème avec l'API Key")
    else:
        for model in models:
            if 'generateContent' in model.supported_generation_methods:
                print(f"✅ {model.name}")
                print(f"   Display: {model.display_name}")
                print(f"   Description: {model.description[:80]}...")
                print()
        
except Exception as e:
    print(f"❌ ERREUR: {e}")
    print("\n⚠️ Vérifiez votre API Key!")

print("\n" + "=" * 60)
print("🧪 TEST DE GÉNÉRATION")
print("=" * 60)

# Test avec différents formats
test_models = [
    "gemini-1.5-flash-latest",
    "gemini-1.5-flash",
    "gemini-1.5-pro-latest", 
    "gemini-1.5-pro",
    "gemini-pro",
    "models/gemini-1.5-flash",
    "models/gemini-pro"
]

for model_name in test_models:
    try:
        print(f"\n🧪 Test: {model_name}...", end=" ")
        model = genai.GenerativeModel(model_name)
        response = model.generate_content("Dis bonjour")
        print(f"✅ FONCTIONNE!")
        print(f"   Réponse: {response.text[:50]}...")
        print(f"\n🎯 UTILISEZ CE MODÈLE: '{model_name}'")
        break
    except Exception as e:
        print(f"❌ Erreur: {str(e)[:100]}")