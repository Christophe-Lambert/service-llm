from flask import Flask, request, jsonify
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM

app = Flask(__name__)

# Charger le modèle pré-entraîné et le tokenizer
# Remplacez ceci par le chemin du modèle monté
model_name = "/app/models"  # Monté via le volume dans le conteneur
tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name,
                                         load_in_8bit=True,       # Ou load_in_4bit=True pour 4-bit
                                         low_cpu_mem_usage=True   # Réduit la mémoire temporaire
                                         )

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()  # Récupérer les données JSON envoyées dans la requête
    text_input = data["text"]  # Extraire le texte de la requête

    # Tokeniser le texte d'entrée
    inputs = tokenizer(text_input, return_tensors="pt")

    # Générer une réponse à l'aide du modèle
    with torch.no_grad():
        outputs = model.generate(inputs["input_ids"], max_length=50)

    # Décoder la réponse générée
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return jsonify({"response": generated_text})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
