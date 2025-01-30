import pandas as pd
from textblob import TextBlob
from textblob.sentiments import PatternAnalyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Função para calcular o sentimento combinado
def calcular_sentimento(texto):
    # TextBlob configurado para português
    blob = TextBlob(texto, analyzer=PatternAnalyzer())
    sentimento_textblob = blob.sentiment[0]  # Pega o score de polaridade (-1 a 1)

    # VADER
    analisador_vader = SentimentIntensityAnalyzer()
    sentimento_vader = analisador_vader.polarity_scores(texto)['compound']  # Também retorna entre -1 e 1

    # Debug: Mostrar valores individuais
    print(f"\nTexto: {texto}")
    print(f"TextBlob: {sentimento_textblob:.3f}, VADER: {sentimento_vader:.3f}")

    # Ajustar pesos para melhorar a precisão
    sentimento_combinado = (0.3 * sentimento_textblob) + (0.7 * sentimento_vader)

    # Definir classificação
    if sentimento_combinado > 0.1:  # Usei 0.1 como margem para evitar neutros errados
        return "Positivo"
    elif sentimento_combinado < -0.1:
        return "Negativo"
    else:
        return "Neutro"

# Carregar o arquivo CSV
df = pd.read_csv('arquivo_exemplo.csv')

# Aplicar a função a cada comentário
df['sentimento'] = df['comentario'].apply(calcular_sentimento)

# Salvar o DataFrame atualizado em um novo arquivo CSV
df.to_csv('comentarios_classificados.csv', index=False)

print("\nClassificação concluída e salva em 'comentarios_classificados.csv'.")
