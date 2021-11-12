# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 10:09:55 2021

@author: Menahem Borges Rodrigues \ מנחם בורג'ס רודריגס
"""
import numpy as np
import math
from sklearn.neighbors.kde import KernelDensity
from scipy import stats

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
    X_train_0,Y_train_0=X_treino[Y_treino==0,:],Y_treino[Y_treino==0] #variaveis treino X class 0 #varaiveis resposta ==0
    X_train_1,Y_train_1=X_treino[Y_treino==1,:],Y_treino[Y_treino==1] #varaiveis resposta == 1 #variaveis treino X class 1

    feats=[i for i in range(np.shape(X_treino)[1])] #definir lista numero de variveis 
    KDE_train0={} #dicionario com o value kde para cada feautures key cls==0
    KDE_train1={} #dicionario com o value kde para cada feautures key cls==1
    #fazer ciclo para estimar a densidade de cada varivel para ambas as classes
    for feat in feats:
        #achar densidade atraves da KDE para class==0
        kde_t0 = KernelDensity(bw, kernel='gaussian')
        kde_t0.fit(X_train_0[:,[feat]],Y_train_0)
        KDE_train0[feat] = (kde_t0)  
        #achar densidade atraves da KDE para class==1        
        kde_t1 = KernelDensity(bw, kernel='gaussian')
        kde_t1.fit(X_train_1[:,[feat]],Y_train_1)
        KDE_train1[feat] = (kde_t1)  
    return KDE_train0,KDE_train1

#dict_train0 e train1 são o output da activate kde
def prever(x,dic_train0,dic_train1,priori0,priori1):
    '''recebe uma nova entrada e calcula a probabilidade de pertencer a uma deteerminada
    class através da more likelihood - ou seja encontrar o argumento y que maximiza a probabilidade
    devolve array com a previsao de class para o input x
    '''
    p_feats_0=[]
    p_feats_1=[]
    #iterar sobre as keys dos dic fit_kde e calcular a prob like para cada variavel
    for i in dic_train0.keys():
        pred_0=dic_train0[i].score_samples(x[:,[i]]) #calcualr o log likelihood cls0
        pred_1=dic_train1[i].score_samples(x[:,[i]]) #calcualr o log likelihood cls1
        p_feats_0.append(pred_0)
        p_feats_1.append(pred_1)    
    #somatorio de probabilidades da entrada para cada varaivel das diferentes features acordo cls 0 e 1
    soma_prob0=np.sum(p_feats_0,axis=0)
    soma_prob1=np.sum(p_feats_1,axis=0)  
    #verificar argmax y class 
    calc_0 = soma_prob0+math.log(priori0) #probabilidade de ser class 0
    calc_1 = soma_prob1+math.log(priori1) #prob ser class 1
    previsao=calc_0-calc_1
    previsao=np.where(previsao>=0,0,1)
    return previsao
     
    
# funcao avalaviadora de classificadores algoritmo mcnemar

def macnemar(NBKDE,NB,SVM,Y_TESTE):
    '''recebe arrays de calssificacao do XTEST dos varios classifcadores
    através de  statistic class1/class2= (Yes/No - No/Yes)^2 / (Yes/No + No/Yes)
    retorna valor KDE VS NB, KDE VS SVM, NB VS SVM'''
    #verificar diferença (boolean) entre classificadores e y_test
    bool_KDE=NBKDE==Y_TESTE
    bool_NB=NB==Y_TESTE
    bool_SVM=SVM==Y_TESTE
    #TABELA NBKDE vs NB
    yes_no_prep=bool_KDE.astype(int)-bool_NB.astype(int)
    yes_no=np.sum(np.where(yes_no_prep==1,1,0))
    no_yes_prep=bool_NB.astype(int)-bool_KDE.astype(int)
    no_yes=np.sum(np.where(no_yes_prep==1,1,0))
    KDE_NB=((yes_no-no_yes)**2)/(yes_no+no_yes)
    pKN=(1-stats.chi2.cdf(KDE_NB,1))  
    #TABELA NBKDE vs SVM
    yes_no_prep2=bool_KDE.astype(int)-bool_SVM.astype(int)
    yes_no2=np.sum(np.where(yes_no_prep2==1,1,0))
    no_yes_prep2=bool_SVM.astype(int)-bool_KDE.astype(int)
    no_yes2=np.sum(np.where(no_yes_prep2==1,1,0))
    KDE_SVM=((yes_no2-no_yes2)**2)/(yes_no2+no_yes2)
    pKS=(1-stats.chi2.cdf(KDE_SVM,1))    
    #TABELA NB VS SVM
    yes_no_prep3=bool_NB.astype(int)-bool_SVM.astype(int)
    yes_no3=np.sum(np.where(yes_no_prep3==1,1,0))
    no_yes_prep3=bool_SVM.astype(int)-bool_NB.astype(int)
    no_yes3=np.sum(np.where(no_yes_prep3==1,1,0))
    NB_SVM=((yes_no3-no_yes3)**2)/(yes_no3+no_yes3)
    pNS=(1-stats.chi2.cdf(NB_SVM,1))      
    return (KDE_NB,pKN),(KDE_SVM,pKS),(NB_SVM,pNS)
