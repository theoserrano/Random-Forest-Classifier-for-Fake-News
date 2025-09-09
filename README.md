# 📰 Classificador de Fake News com Machine Learning

Este projeto é um modelo de Machine Learning desenvolvido em Python para classificar notícias como verdadeiras ou falsas, utilizando um dataset com mais de 20.000 artigos. O objetivo é demonstrar um fluxo completo de um projeto de ciência de dados, desde a preparação dos dados até o treinamento e avaliação de um modelo classificador.

## 🎯 Objetivo

O principal objetivo é aplicar técnicas de pré-processamento de dados em diferentes tipos de features (texto, categóricas e numéricas) e utilizar o algoritmo **Random Forest** para realizar a classificação, medindo sua eficácia através da acurácia.

## 🛠️ Tecnologias Utilizadas

* **Python 3.x**
* **Pandas:** Para manipulação e análise dos dados.
* **Scikit-learn:** Para o pipeline de pré-processamento, modelo e métricas de avaliação.

## ⚙️ Funcionalidades

O script `main.py` executa as seguintes ações:
1.  **Carregamento dos Dados:** Lê o arquivo `FakeNewsNet.csv` usando Pandas.
2.  **Divisão do Dataset:** Separa os dados em conjuntos de treino e teste.
3.  **Pipeline de Pré-processamento:** Constrói e aplica um `ColumnTransformer` para tratar as features:
    * `title` (texto): É vetorizada usando `TfidfVectorizer`.
    * `source_domain` (categórica): É codificada com `OneHotEncoder`.
    * `tweet_num` (numérica): É normalizada com `StandardScaler`.
4.  **Treinamento do Modelo:** Treina um classificador `RandomForestClassifier` com os dados processados.
5.  **Avaliação:** Realiza previsões no conjunto de teste e calcula a acurácia do modelo.
