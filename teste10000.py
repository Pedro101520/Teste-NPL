import pandas as pd
import spacy
import nltk
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from deep_translator import GoogleTranslator
from nltk.corpus import stopwords

# Baixar stopwords do NLTK
nltk.download('stopwords')
stopwords_en = set(stopwords.words('english'))

# Carregar modelo Spacy para processamento de texto
nlp = spacy.load("en_core_web_sm")

# Função para traduzir o texto para inglês
def traduzir_para_ingles(texto):
    try:
        return GoogleTranslator(source='auto', target='en').translate(texto)
    except:
        return texto  # Se a tradução falhar, retorna o texto original

# Função para remover stopwords
def remover_stopwords(texto):
    palavras = texto.split()
    palavras_filtradas = [palavra for palavra in palavras if palavra.lower() not in stopwords_en]
    return ' '.join(palavras_filtradas)

# Função para calcular o sentimento do texto combinando TextBlob, VADER e Spacy
def calcular_sentimento(texto):
    if not texto.strip():  # Se o texto estiver vazio, retorna "Neutro"
        return "Neutro"

    # Traduzir para inglês
    texto_traduzido = traduzir_para_ingles(texto)

    # Remover stopwords
    texto_sem_stopwords = remover_stopwords(texto_traduzido)

    # Análise com TextBlob
    blob = TextBlob(texto_sem_stopwords)
    sentimento_textblob = blob.sentiment.polarity  # Valor entre -1 e 1

    # Análise com VADER
    analisador_vader = SentimentIntensityAnalyzer()
    sentimento_vader = analisador_vader.polarity_scores(texto_sem_stopwords)['compound']

    # Processamento com Spacy (extração de entidades e relevância)
    doc = nlp(texto_sem_stopwords)
    relevancia_spacy = sum([token.sentiment for token in doc if token.is_alpha]) / (len(doc) + 1)

    # Cálculo ponderado dos sentimentos
    sentimento_combinado = (0.4 * sentimento_textblob) + (0.4 * sentimento_vader) + (0.2 * relevancia_spacy)

    # Classificação final com base na média ponderada
    if sentimento_combinado > 0.1:
        return "Positivo"
    elif sentimento_combinado < -0.1:
        return "Negativo"
    else:
        return "Neutro"

# Carregar os comentários do arquivo CSV
df = pd.read_csv('arquivo_exemplo.csv')

# Aplicar a função a cada comentário
df['sentimento'] = df['comentario'].apply(calcular_sentimento)

# Salvar os resultados em um novo CSV
df.to_csv('comentarios_classificados.csv', index=False)

print("Classificação concluída e salva em 'comentarios_classificados.csv'.")
