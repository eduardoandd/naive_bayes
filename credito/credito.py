import pandas as pd
import shutil
import pickle
from sklearn.naive_bayes import  GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report
from yellowbrick.classifier import ConfusionMatrix

# arquivo= 'C:/Users/edubo/Desktop/I.A/pre-processamento/credit.pkl'
# destino='C:/Users/edubo/Desktop/I.A/naive_bayes/credito/credit.pkl'
# shutil.copy(arquivo,destino)

with open('credit.pkl', 'rb') as f:
    X_credit_treinamento,y_credit_treinamento,x_credit_teste,y_credit_teste=pickle.load(f)

nave_credit_data= GaussianNB()
nave_credit_data.fit(X_credit_treinamento,y_credit_treinamento)
previsao=nave_credit_data.predict(x_credit_teste)

#métricas
accuracy_score(y_credit_teste,previsao) #porcentagem de acerto
confusion_matrix(y_credit_teste,previsao)
cm=ConfusionMatrix(nave_credit_data)
cm.fit(X_credit_treinamento,y_credit_treinamento)
cm.score(x_credit_teste,y_credit_teste)
classification_report(y_credit_teste,previsao)