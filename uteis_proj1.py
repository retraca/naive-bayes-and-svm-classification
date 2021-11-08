# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 10:09:55 2021

@author: Menahem Borges Rodrigues \ מנחם בורג'ס רודריגס
"""
import numpy as np
import math
from sklearn.neighbors.kde import KernelDensity

def load_data(path):
    ''' esta funcao recebe o ficheiro de dados
    retorna um np array'''
    data=[]
    try:
        data=np.loadtxt(path,delimiter='\t') #load dos dados em formato array
        data[:,-1]=data[:,-1].astype((np.int32)) #transforma y em inteiros
    except Exception as inst:
        print(type(inst),inst) 
    return data

def standartize(data):
    '''recebe array dados e faz a sua standartizacao'''
    means=np.mean(data[:,:-1]) # media todas as coluna excepto y
    stdevs=np.std(data[:,:-1]) # std todas as coluna excepto y
    data[:,:-1]=(data[:,:-1]-means)/stdevs #aplica a standartizacao a x
    return data

def misturar_data(data):
    ranks=np.arange(data.shape[0])
    np.random.shuffle(ranks) #espalha aleatoria os indeces do arrays
    train=data[ranks,:]
    return train

############################################################################

# funcoes para implementar o KDE

def activate_kde(X_treino,Y_treino,bw):
    '''funcao recebe data treino e parametro de alrgura de banda para bw
    retorna dicionarios com as variaveis como key e as suas densidades como
    values'''
    X_train_0=X_treino[Y_treino==0,:] #variaveis treino X class 0
    Y_train_0=Y_treino[Y_treino==0,:] #varaiveis resposta ==0
    X_train_1=X_treino[Y_treino==1,:] #variaveis treino X class 1
    Y_train_1=Y_treino[Y_treino==1,:] #varaiveis resposta == 1
    feats=[i for i in range(X_treino.shape[1])] #definir lista numero de variveis 
    KDE_train0={} #dicionario com o value kde para cada feautures key cls==0
    KDE_train1={} #dicionario com o value kde para cada feautures key cls==1
    #fazer ciclo para estimar a densidade de cada varivel para ambas as classes
    for feat in feats:
        #achar densidade atraves da KDE para class==0
        kde_t0 = KernelDensity(bw, kernel='gaussian')
        kde_t0.fit(X_train_0[:,feat])
        KDE_train0[feat] = (kde_t0)  
        #achar densidade atraves da KDE para class==1        
        kde_t1 = KernelDensity(bw, kernel='gaussian')
        kde_t1.fit(X_train_1[:,feat])
        KDE_train1[feat] = (kde_t1)  
    return KDE_train0,KDE_train1

#dict_train0 e train1 são o output da activate kde
def predict(x,dic_train0,dic_train1,priori0,priori1):
    '''recebe uma nova entrada e calcula a probabilidade de pertencer a uma deteerminada
    class através da more likelihood - ou seja encontrar o argumento y que maximiza a probabilidade
    '''
    p_feats_0=[]
    p_feats_1=[]
    #iterar sobre as keys dos dic fit_kde e calcular a prob like para cada variavel
    for i in dic_train0.keys():
        pred_0=dic_train0[i].score_sample(x[:,[i]]) #calcualr o log likelihood cls0
        pred_1=dic_train1[i].score_sample(x[:,[i]]) #calcualr o log likelihood cls1
        p_feats_0.append(pred_0)
        p_feats_1.append(pred_1)    
    #somatorio de probabilidades da entrada para cada varaivel das diferentes features acordo cls 0 e 1
    soma_prob0=np.sum(p_feats_0,axis=0)
    soma_prob1=np.sum(p_feats_1,axis=0)    
    calcul_p0=[]
    calcul_p1=[]   
    #ciclo sobre soma_prob
    for i in range(len(soma_prob0)):
        calc_0 = math.log(priori0) + soma_prob0[i]
        calc_1 = math.log(priori1) + soma_prob1[i]
        calcul_p0.append(calc_0)
        calcul_p1.append(calc_1)  
        
     
    
    
    