import pandas as pd
import shutil
import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report
from yellowbrick.classifier import ConfusionMatrix

arquivo= 'C:/Users/edubo/Desktop/I.A/pre-processamento/census.pkl'
destino='C:/Users/edubo/Desktop/I.A/naive_bayes/census/census.pkl'
shutil.copy(arquivo,destino)

with open('census.pkl', 'rb') as f:
    X_treinamento,y_treinamento,X_teste,y_teste=pickle.load(f)

nave_census_data= GaussianNB()
nave_census_data.fit(X_treinamento,y_treinamento)
previsao = nave_census_data.predict(X_teste)

accuracy_score(y_teste,previsao)
cm=ConfusionMatrix(nave_census_data)
cm.fit(X_treinamento,y_treinamento)
cm.score(X_teste,y_teste)