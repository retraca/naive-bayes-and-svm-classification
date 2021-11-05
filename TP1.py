# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 11:33:01 2021

@author: Menahem Borges Rodrigues \ מנחם בורג'ס רודריגס
"""

''' Projecto 1 - Machine Learning |||| Naive bayes & SVM classification'''
from uteis_proj1 import load_data,standartize,misturar_data
import numpy as np
from sklearn.model_selection import StratifiedKFoldi
from sklearn.naive_bayes import GaussianNB

'''preparacao dos dados'''

#import data
data_tr=load_data('TP1_train.tsv') #criar array treino 1246*(4+1)
data_test=load_data('TP1_test.tsv') #criar array test  1247*(4+1)

def preparar_dados(data_treino,data_teste):
    '''recebe como input dados de treino e dados teste
    retorna 4 arrays X, Y_treino e X Y_Teste estandartizados e espalhados'''
    data_tr=standartize(data_treino) #dados treino standartizados
    data_test=standartize(data_teste) #dados test standartizadso
    #shuffle data
    data_tr=misturar_data(data_tr) #data treino com random 
    #criar arrays para variaveis X e e variavel resposta Y
    X_train=data_tr[:,:-1] # treino todas as colunas excepto ultima Y
    Y_train=data_tr[:,-1] #treino ultima col
    X_test=data_test[:,:-1] #test todas as colunas excepto ultima Y
    Y_test=data_test[:,-1] #test ultima col
    return X_train,Y_train,X_test,Y_test

X_train,Y_train,X_test,Y_test=preparar_dados(data_tr, data_test)

#implementar classificador Naive Bayes kde com ajuste de bandwith
# considerando desconhecermos a distribuicao dos dados utilizamos o kde para conseguirmos obter as probabilidades de classificacao

# cross validation através Startifiedkfold para dividir as variaveis respsota de forma propocional
kf = StratifiedKFold(n_splits = 5) #divisão será estratificada data será dividida em 5 folds treinará  em 4 e validará em 1
NB_kde_Te = [] #erro treino
NB_kde_Ve = [] #erro validacao
H=np.arange(0.02,0.6,0.02) #definir array bandwith 

for h in H: #iterar sobrebandwith e verificar errors para cada b
    train_error_nv=0
    valid_error_nv=0 #inicializar erros a 0; será feita méia no final sum error/fold
    for train_idx,valid_idx in kf.split(Y_train,Y_train): # iterar sobre 5 vezes sobre as diferentes divisoes
        #fit de model NB KDE e calcula o 1-score associado (fracao classificaçoes incorrectas)  
        
plt.scatter()        
    
