
"""
Created on Tue Oct 26 11:33:01 2021

@author: Menahem Borges Rodrigues \ מנחם בורג'ס רודריגס
"""

''' Projecto 1 - Machine Learning |||| Naive bayes & SVM classification'''
from McNemar import McNemar
from uteis_proj1 import load_data, standartize,misturar_data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from NaiveBayesClassifier import NaiveBayes
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

print(X_train)
#probabilidade priori
#priori_0=(X_train[Y_train==0,:].shape(0))/X_train.shape(0) #numero de elementos da respectiva clss / numero total
#priori_1=1-priori_0 #como são apenas 2 classes utilizamos 1 - prob contraria
#implementar classificador Naive Bayes kde com ajuste de bandwith
# considerando desconhecermos a distribuicao dos dados utilizamos o kde para conseguirmos obter as probabilidades de classificacao

# cross validation através Startifiedkfold para dividir as variaveis respsota de forma propocional
kf = StratifiedKFold(n_splits = 5) #divisão será estratificada data será dividida em 5 5 treinará  em 4 e validará em 1
feats=[1,2,3,4]
NB_kde_te = [] #erro treino
NB_kde_ve = [] #erro validacao
H=np.arange(0.02,0.6,0.02) #definir array bandwith 

for h in H: #iterar sobrebandwith e verificar errors para cada b
    train_error_nv=0
    valid_error_nv=0 #inicializar erros a 0; será feita méia no final sum error/fold
    for train_idx,valid_idx in kf.split(Y_train,Y_train): # iterar sobre 5 vezes sobre as diferentes divisoes
        #fit de model NB KDE e calcula o 1-score associado (fracao classificaçoes incorrectas)  
        X_tk,Y_tk,X_vk,Y_vk=X_train[train_idx],Y_train[train_idx],X_train[valid_idx],Y_train[valid_idx]

        nb = NaiveBayes(h,feats)

        nb.fit(X_tk,Y_tk)
        nb.predict(X_tk,Y_tk)
        t = nb.score(Y_tk)
        
        nb.predict(X_vk,Y_vk)
        v = nb.score(Y_vk)
        
        train_error_nv += t
        valid_error_nv += v

    
        """
        kde_0,kde_1=activate_kde(X_tk,Y_tk,h)
        y_prev_train=prever(X_tk,kde_0,kde_1,priori_0,priori_1) #prev class training set
        t_e=1-accuracy_score(Y_tk, y_prev_train) #calc err para training set
        #repetir proc para prev dados valid
        y_prev_valid=prever(X_vk,kde_0,kde_1,priori_0,priori_1) #prev class valid set
        t_val=1-accuracy_score(Y_vk, y_prev_valid) #calc err para valid set
        train_error_nv+=t_e
        valid_error_nv+=t_val
        NB_kde_te.append(train_error_nv/5)
        NB_kde_ve.append(valid_error_nv/5)
        """
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
bw_star = H[np.argmin(NB_kde_ve)]
nb = NaiveBayes(bw_star,feats)
nb.fit(X_train,Y_train)
pred_NB = nb.predict(X_test,Y_test)
NB_true_error = nb.score(Y_test)
print("NB True Error: ",np.round(NB_true_error,5)) #verificar score atraves da accuracy do sklearn

##################################################################################
# Implementar Naive Bayes Gaussian do SKlearn

clf_GNB = GaussianNB()
clf_GNB.fit(X_train, Y_train)
pred_GNB = clf_GNB.predict(X_test)
train_error_GNB = 1-clf_GNB.score(X_train, Y_train)
GNB_te = 1-clf_GNB.score(X_test, Y_test)
print("\nGaussianNB Training Error: ",np.round(train_error_GNB,5),\
      "\nGaussianNB True Error: ",np.round(GNB_te,5))

##############Determination of the gamma on the training set (SVM)############
###Cross Validation
train_error_SVM = []
valid_error_SVM = []
gamma = np.arange(0.2,6.1,0.2)
  
for g in gamma:
    
    t_error_SVM = v_error_SVM = 0
    
    for train_idx, valid_idx in kf.split(Y_train,Y_train):
        
        xt_set = X_train[train_idx]
        yt_set = Y_train[train_idx]
        xv_set = X_train[valid_idx]
        yv_set = Y_train[valid_idx]
        
        clf = SVC(C = 1 , kernel = "rbf", gamma = g)
        clf.fit(xt_set,yt_set)        
        
        t = 1-clf.score(xt_set,yt_set)
        t_error_SVM += t
        
        v = 1-clf.score(xv_set,yv_set)
        v_error_SVM += v
    
    train_error_SVM.append(t_error_SVM/5)
    valid_error_SVM.append(v_error_SVM/5)
    #print("\nGamma: ",g,':', t_error_SVM/5,'\t',v_error_SVM/5)

plt.plot(gamma, train_error_SVM,'b',label='Training Error')
plt.plot(gamma, valid_error_SVM, 'r',label='Validation Error')
plt.title('SVM: Training Error vs Validation Error') 
plt.xlabel('Gamma')
plt.ylabel('Error')
plt.legend(loc='upper right',frameon=False)
plt.savefig('SVM.png', dpi=300)
plt.close()

print("\nBest gamma =",np.round(gamma[np.argmin(valid_error_SVM)],2))
print("SVM Training Error: ",np.round\
      (train_error_SVM[np.argmin(valid_error_SVM)],5),
   "\nSVM Validation Error: ",np.round(min(valid_error_SVM),5))

clf_SVM = SVC(C=1.0 , kernel = "rbf", gamma = gamma[np.argmin(valid_error_SVM)])
clf_SVM.fit(X_train,Y_train) 
pred_SVM = clf_SVM.predict(X_test)
SVM_te = 1-clf_SVM.score(X_test,Y_test)
print("SVM True Error: ",np.round(SVM_te,5))


########################Comparing classifiers#################################
n = len(Y_test) #number of observations

###Number of errors in each classifier test
##Naive Bayes
E_NB = n * NB_true_error
sigma_NB = np.sqrt(NB_true_error*(1-NB_true_error))
#print(E_NB)

##GaussianNB
E_GNB = n * GNB_te
sigma_GNB = np.sqrt(GNB_te*(1-GNB_te))
#print(E_GNB)

##SVM
E_SVM = n * SVM_te
sigma_SVM = np.sqrt(SVM_te*(1-SVM_te))
#print(E_SVM)

###For an alpha of 0.05, the IC's are given by
IC_NB = [E_NB - 1.96*sigma_NB, E_NB + 1.96*sigma_NB]
print('\nIC to Naive Bayes: ',np.round(IC_NB,5))

IC_GNB = [E_GNB - 1.96*sigma_GNB, (E_GNB + 1.96*sigma_GNB)]
print('IC to Gaussian Naive Bayes: ',np.round(IC_GNB,5))

IC_SVM = [E_SVM - 1.96*sigma_SVM, (E_SVM + 1.96*sigma_SVM)]
print('IC to SVM: ',np.round(IC_SVM,5))

###Mcnemar test
q,w,e = McNemar(pred_NB,pred_GNB,pred_SVM,Y_test)

print("\nMcnemar's Test:")
print('NB_vs_GNB = ',np.round(q,5),'\nNB_vs_SVM = ',\
      np.round(w,5),'\nGNB_vs_SVM = ',np.round(e,5))

    
