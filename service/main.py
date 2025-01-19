from transformers import LlamaForCausalLM, LlamaTokenizer
import csv

# Charger le modèle Llama 2 et le tokenizer
#model_name = "meta-llama/Llama-2-7b"  # Remplacez par le chemin local ou le modèle téléchargé
model_name = "D:/Projects/test-angular/Llama-2-7b-chat-hf"

tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name)

# Fonction pour interroger le modèle Llama 2
def query_llama2(question):
    # Tokenisation de la question
    inputs = tokenizer(question, return_tensors="pt")

    # Générer une réponse
    output = model.generate(**inputs, max_length=100)

    # Décoder et retourner la réponse générée
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# Lire un fichier CSV et l'analyser avec Llama 2
def read_csv_and_analyze(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        csv_text = "\n".join([",".join(row) for row in reader])

        # Question pour Llama 2
        question = f"Voici un extrait de fichier CSV : {csv_text}\n\nQuels sont les numéros de téléphone ?"

        # Interroger le modèle avec la question
        result = query_llama2(question)
        print("Résultat extrait : ", result)

# Exemple d'utilisation
read_csv_and_analyze('sample.csv')
