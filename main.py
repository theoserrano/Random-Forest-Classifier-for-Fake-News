import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

newsdataset = pd.read_csv('FakeNewsNet.csv')

x_newsdataset = newsdataset[['title', 'source_domain', 'tweet_num']] 
y_newsdataset = newsdataset[['real']]

x_newsdataset_treinamento, x_newsdataset_teste, y_newsdataset_treinamento, y_newsdataset_teste = train_test_split(x_newsdataset, y_newsdataset, test_size=0.25)

preprocessor = ColumnTransformer(
    transformers=[
        ('title_tfidf', TfidfVectorizer(stop_words='english', max_features=5000), 'title'),
        ('source_ohe', OneHotEncoder(handle_unknown='ignore'), ['source_domain']),
        ('tweet_scaler', StandardScaler(), ['tweet_num'])
    ])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators= 80, criterion= 'entropy'))
])

pipeline.fit(x_newsdataset_treinamento, y_newsdataset_treinamento)

previsoes = pipeline.predict(x_newsdataset_teste)

from sklearn.metrics import accuracy_score

print(accuracy_score(y_newsdataset_teste, previsoes))