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
from sklearn.metrics import accuracy_score

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

#probabilidade priori
priori_0=(X_train[Y_train==0,:].shape(0))/X_train.shape(0) #numero de elementos da respectiva clss / numero total
priori_1=1-priori_0 #como são apenas 2 classes utilizamos 1 - prob contraria
#implementar classificador Naive Bayes kde com ajuste de bandwith
# considerando desconhecermos a distribuicao dos dados utilizamos o kde para conseguirmos obter as probabilidades de classificacao

# cross validation através Startifiedkfold para dividir as variaveis respsota de forma propocional
kf = StratifiedKFold(n_splits = 5) #divisão será estratificada data será dividida em 5 folds treinará  em 4 e validará em 1
NB_kde_te = [] #erro treino
NB_kde_ve = [] #erro validacao
H=np.arange(0.02,0.6,0.02) #definir array bandwith 
for h in H: #iterar sobrebandwith e verificar errors para cada b
    train_error_nv=0
    valid_error_nv=0 #inicializar erros a 0; será feita méia no final sum error/fold
    for train_id,valid_id in kf.split(Y_train,Y_train): # iterar sobre 5 vezes sobre as diferentes divisoes
        #fit de model NB KDE e calcula o 1-score associado (fracao classificaçoes incorrectas)  
        X_tk,Y_tk,X_vk,Y_vk=X_train[train_id],Y_train[train_id],X_train[valid_id],Y_train[valid_id]
        kde_0,kde_1=activate_kde(X_tk,Y_tk,h)
        y_prev_train=prever(X_tk,kde_0,kde_1,priori_0,priori_1) #prev class training set
        t_e=1-accuracy_score(y_tk, y_prev_train) #calc err para training set
        #repetir proc para prev dados valid
        y_prev_valid=prever(X_vk,kde_0,kde_1,priori_0,priori_1) #prev class valid set
        t_val=1-accuracy_score(y_vk, y_prev_valid) #calc err para valid set
        train_error_nv+=t_e
        valid_error_nv+=t_val
      NB_kde_te.append(train_error_nv/5)
      NB_kde_ve.append(valid_error_nv/5)
      
#################################################################################
# fazer plot do H (bandwiths eixo x) e erros no y para verificar evolucao erros com h
plt.figure()
plt.title('Naive Bayes KDE - Training vs Validation Error (bandwiths)')
plt.plot(H,NB_kde_te,'blue',label='training_err') #plot erros treino acordo h
plt.plot(H,NB_kde_ve,'red',label='validation_err') #plot erros valid acordo h
plt.xlabel('bandwidth')
plt.ylabel('error')
plt.savefig('NB.png',dpi=250) #gravar com nome requerido 
plt.close()

#imprimir melhor h (indice valor min erros validacao
print ('melhor valor bandwidth: ', H[np.argmin(NB_kde_ve)], ' com um erro de validacao de: ', min(NB_kde_ve))
#calcuclar o erro real da nossa implementacao com o melhor bandwidth
h_min=H[np.argmin(NB_kde_ve)]
kde_0,kde_1=activate_kde(X_train,Y_train,h_min) # achar densidades das features com o h:min
y_prev_test=prever(X_test,kde_0,kde_1,priori_0,priori_1) #prever com classes do X test com os kde 
test_score=1-accuracy_score(Y_test, y_prev_test) #verificar score atraves da accuracy do sklearn

##################################################################################
# Implementar Naive Bayes Gaussian do SKlearn



    
