import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
import joblib
from nltk import tokenize
import re

# Função para limpar o texto
def limpar_texto(texto):
    texto = re.sub(r"http\S+|www\S+|https\S+", '', texto)
    texto = re.sub(r"@\w+", '', texto)
    texto = re.sub(r"[^\w\s]", '', texto, flags=re.UNICODE)
    texto = re.sub(r"\d+", '', texto)
    texto = texto.strip()
    return texto

# Carregando os dados
df = pd.read_csv('https://raw.githubusercontent.com/alura-cursos/nlp_analise_sentimento/refs/heads/main/Dados/dataset_avaliacoes.csv')

# Aplicando a limpeza do texto
df['tratamento_0'] = df['avaliacao'].apply(limpar_texto)

# Definindo stopwords
import nltk
nltk.download("stopwords")
palavras_irrelevantes = nltk.corpus.stopwords.words('portuguese')

# Removendo palavras irrelevantes
token_espaco = tokenize.WhitespaceTokenizer()
frase_processada = []
for opiniao in df['tratamento_0']:
    palavras_texto = token_espaco.tokenize(opiniao)
    nova_frase = [palavra for palavra in palavras_texto if palavra not in palavras_irrelevantes]
    frase_processada.append(' '.join(nova_frase))
df['tratamento_1'] = frase_processada

# Remove pontuação
token_pontuacao = tokenize.WordPunctTokenizer()
frase_processada = []
for opiniao in df['tratamento_1']:
    palavras_texto = token_pontuacao.tokenize(opiniao)
    nova_frase = [palavra for palavra in palavras_texto if palavra.isalpha() and palavra not in palavras_irrelevantes]
    frase_processada.append(' '.join(nova_frase))
df['tratamento_2'] = frase_processada

# Remove acentuação
import unidecode
sem_acentos = [unidecode.unidecode(texto) for texto in df['tratamento_2']]
df['tratamento_3'] = sem_acentos

# Remover mais palavras irrelevantes
frase_processada = []
for opiniao in df['tratamento_3']:
    palavras_texto = token_pontuacao.tokenize(opiniao)
    nova_frase = [palavra for palavra in palavras_texto if palavra not in palavras_irrelevantes]
    frase_processada.append(' '.join(nova_frase))
df['tratamento_4'] = frase_processada

# Convertendo para minúsculas e removendo palavras irrelevantes
frase_processada = []
for opiniao in df['tratamento_4']:
    opiniao = opiniao.lower()
    palavras_texto = token_pontuacao.tokenize(opiniao)
    nova_frase = [palavra for palavra in palavras_texto if palavra not in palavras_irrelevantes]
    frase_processada.append(' '.join(nova_frase))
df['tratamento_5'] = frase_processada

# Ajuste dos hiperparâmetros usando GridSearchCV
# TfidfVectorizer com ngram_range de 1 a 2 (unigrams e bigrams)
tfidf_1000 = TfidfVectorizer(lowercase=False, max_features=1000, ngram_range=(1, 2))

# Vetorizando os textos
vetor_tfidf = tfidf_1000.fit_transform(df['tratamento_5'])

# Definindo os hiperparâmetros para o modelo de regressão logística
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],  # Valores diferentes para o parâmetro de regularização
    'penalty': ['l2']  # Apenas L2 neste caso
}

# Usando GridSearchCV com 5 folds de validação cruzada
grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring='accuracy')

# Ajustando o modelo
grid_search.fit(vetor_tfidf, df['sentimento'])

# Mostrando os melhores parâmetros encontrados
print(f'Melhor acurácia: {grid_search.best_score_ * 100:.2f}%')
print(f'Melhores hiperparâmetros: {grid_search.best_params_}')

# Treinando o modelo final com os melhores parâmetros
melhor_modelo = grid_search.best_estimator_

# Salvando o modelo ajustado
joblib.dump(melhor_modelo, 'modelo_regressao_logistica.pkl')
joblib.dump(tfidf_1000, 'tfidf_vectorizer.pkl')

print("Terminou a execução")
