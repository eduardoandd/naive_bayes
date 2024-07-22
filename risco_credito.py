import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle
from sklearn.naive_bayes import GaussianNB

base_risco_credito=pd.read_csv('risco_credito.csv')

X= base_risco_credito.iloc[:,0:4].values
y=base_risco_credito.iloc[:,-1].values
label_encoder=LabelEncoder()

for i in range(X.shape[1]):
    if X[:,i].dtype == 'object':
        X[:,i] = label_encoder.fit_transform(X[:,i])

with open('risco_credito.pkl','wb') as f:
    pickle.dump([X, y], f)

naive_risco_credito = GaussianNB()
naive_risco_credito.fit(X,y) # gera a tabela de probabilidade

#historia boa(0); divida alta(0); garantia nenhuma(1); >35(2)
previsao = naive_risco_credito.predict([[0,0,1,2],[1,2,2,0]])

