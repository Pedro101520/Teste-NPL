from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# Supondo que você já tenha o seu DataFrame 'df' com os dados
# df['tratamento_5'] são os comentários e df['sentimento'] são as classes (positivo, negativo, neutro)

# Parâmetros de ajuste para o TfidfVectorizer
parametros_tfidf = {
    'max_features': [1000, 5000, 10000],  # Número de features
    'ngram_range': [(1, 1), (1, 2), (1, 3)],  # Unigrams, Bigrams, Trigrams
    'min_df': [0.01, 0.05],  # Frequência mínima do termo
    'max_df': [0.85, 0.95],  # Frequência máxima do termo
    'sublinear_tf': [True, False]  # Aplicar transformação logarítmica
}

# Parâmetros de ajuste para a Regressão Logística
parametros_regressao = {
    'C': [0.001, 0.01, 0.1, 1, 10],  # Regularização
    'solver': ['liblinear', 'saga'],  # Solvers para otimização
    'penalty': ['l1', 'l2'],  # Penalidade (regularização)
    'max_iter': [100, 200, 500],  # Número máximo de iterações
    'multi_class': ['ovr', 'multinomial']  # Estratégia multi-classe
}

# Inicializando o TfidfVectorizer
tfidf = TfidfVectorizer()

# Inicializando o modelo de Regressão Logística
regressao_logistica = LogisticRegression()

# Criando o GridSearchCV para ajustar os parâmetros
grid_search = GridSearchCV(
    estimator=regressao_logistica,
    param_grid={
        'tfidfvectorizer__max_features': parametros_tfidf['max_features'],
        'tfidfvectorizer__ngram_range': parametros_tfidf['ngram_range'],
        'tfidfvectorizer__min_df': parametros_tfidf['min_df'],
        'tfidfvectorizer__max_df': parametros_tfidf['max_df'],
        'tfidfvectorizer__sublinear_tf': parametros_tfidf['sublinear_tf'],
        'logisticregression__C': parametros_regressao['C'],
        'logisticregression__solver': parametros_regressao['solver'],
        'logisticregression__penalty': parametros_regressao['penalty'],
        'logisticregression__max_iter': parametros_regressao['max_iter'],
        'logisticregression__multi_class': parametros_regressao['multi_class']
    },
    cv=5,  # número de folds para validação cruzada
    n_jobs=-1,  # usar todos os núcleos disponíveis para computação paralela
    verbose=1  # para ver mais detalhes do processo
)

# Separando dados de treino e teste
X_treino, X_teste, y_treino, y_teste = train_test_split(df['tratamento_5'], df['sentimento'], random_state=42)

# Ajustando o modelo com o GridSearchCV
grid_search.fit(X_treino, y_treino)

# Exibindo os melhores parâmetros encontrados e a melhor acurácia
print("Melhores parâmetros encontrados: ", grid_search.best_params_)
print("Melhor Acurácia: ", grid_search.best_score_)

# Salvando o melhor modelo e o TfidfVectorizer com os melhores parâmetros
melhor_tfidf = grid_search.best_estimator_.named_steps['tfidfvectorizer']
melhor_modelo = grid_search.best_estimator_.named_steps['logisticregression']

# Salvando os modelos
with open('melhor_tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(melhor_tfidf, f)

with open('melhor_regressao_logistica.pkl', 'wb') as f:
    pickle.dump(melhor_modelo, f)

# Avaliando o modelo no conjunto de teste
acuracia_teste = grid_search.score(X_teste, y_teste)
print(f'Acurácia no conjunto de teste: {acuracia_teste * 100:.2f}%')
