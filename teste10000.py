from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Carregar modelo e tokenizador
modelo = "neuralmind/bert-base-portuguese-cased"
tokenizer = AutoTokenizer.from_pretrained(modelo)
modelo = AutoModelForSequenceClassification.from_pretrained(modelo, num_labels=3)  # Positivo, Neutro, Negativo

def analisar_sentimento(texto):
    """Classifica um texto como positivo, negativo ou neutro usando BERTimbau."""
    tokens = tokenizer(texto, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    # Fazer previsão
    with torch.no_grad():
        output = modelo(**tokens)
    
    # Pegando o resultado
    scores = torch.nn.functional.softmax(output.logits, dim=1)
    labels = ["Negativo", "Neutro", "Positivo"]
    
    # Retornar o sentimento com maior probabilidade
    sentimento = labels[torch.argmax(scores).item()]
    return sentimento

# Teste
comentarios = [
    "O atendimento foi excelente! Adorei a experiência.",
    "O produto é horrível, nunca mais compro.",
    "Foi um serviço normal, nada demais."
]

for comentario in comentarios:
    print(f"Comentário: {comentario} -> Sentimento: {analisar_sentimento(comentario)}")
