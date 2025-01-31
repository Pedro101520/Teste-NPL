import nltk
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Baixar recursos do NLTK (para VADER)
nltk.download('vader_lexicon')

# Função para análise com VADER
def vader_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    score = analyzer.polarity_scores(text)
    # Retorna a classificação com base no score de "compound"
    if score['compound'] >= 0.05:
        return 'positivo'
    elif score['compound'] <= -0.05:
        return 'negativo'
    else:
        return 'neutro'

# Função para análise com TextBlob-Pt
def textblob_sentiment(text):
    blob = TextBlob(text)
    # A polaridade varia de -1 (negativo) a 1 (positivo)
    polarity = blob.sentiment.polarity
    if polarity > 0:
        return 'positivo'
    elif polarity < 0:
        return 'negativo'
    else:
        return 'neutro'

# Função para análise com SentiLex-PT (baseado em um dicionário simples de sentimentos)
# Aqui você precisaria de um dicionário ou lista de palavras com sentimentos pré-definidos
def senti_lex_sentiment(text):
    # Exemplo de lista simples (isso precisa ser expandido para ser eficaz)
    positive_words = ['bom', 'ótimo', 'maravilhoso', 'feliz', 'excelente']
    negative_words = ['ruim', 'péssimo', 'triste', 'horrível', 'detestável']
    
    score = 0
    words = text.lower().split()
    
    for word in words:
        if word in positive_words:
            score += 1
        elif word in negative_words:
            score -= 1
    
    if score > 0:
        return 'positivo'
    elif score < 0:
        return 'negativo'
    else:
        return 'neutro'

# Função de votação ponderada
def weighted_voting(text):
    # Analisadores
    vader_vote = vader_sentiment(text)
    textblob_vote = textblob_sentiment(text)
    senti_lex_vote = senti_lex_sentiment(text)
    
    # Pesos (aqui podem ser ajustados conforme a confiança nos métodos)
    weights = {'positivo': 1, 'negativo': -1, 'neutro': 0}
    
    # Ponderação dos votos
    weighted_votes = {
        'positivo': weights[vader_vote] + weights[textblob_vote] + weights[senti_lex_vote],
        'negativo': weights[vader_vote] + weights[textblob_vote] + weights[senti_lex_vote],
        'neutro': weights[vader_vote] + weights[textblob_vote] + weights[senti_lex_vote]
    }
    
    # Retorna o voto com maior ponderação
    return max(weighted_votes, key=weighted_votes.get)

# Exemplos de comentários
comments = [
    "Este produto é maravilhoso e muito bom!",
    "Não gostei nada desse filme, foi péssimo.",
    "O tempo está ok hoje, nem bom nem ruim.",
    "Adorei o evento, foi excelente!",
    "Não foi um bom dia para mim."
]

# Classificação dos comentários
for comment in comments:
    print(f"Comentário: {comment}")
    print(f"Sentimento: {weighted_voting(comment)}")
    print("-" * 30)
