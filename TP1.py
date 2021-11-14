# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 11:33:01 2021

@author: Menahem Borges Rodrigues \ מנחם בורג'ס רודריגס
"""

''' Projecto 1 - Machine Learning |||| Naive bayes & SVM classification'''
from uteis_proj1 import load_data,standartize,misturar_data,activate_kde,prever,macnemar
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from scipy import stats
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
#print(X_train,Y_train,X_test,Y_test)

#probabilidade priori
priori_0=(np.shape(X_train[Y_train==0,:])[0])/np.shape(X_train)[0] #numero de elementos da respectiva clss / numero total
priori_1=1-priori_0 #como são apenas 2 classes utilizamos 1 - prob contraria
#print(priori_0, priori_1)
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
        t_e=1-accuracy_score(Y_tk, y_prev_train) #calc err para training set
        #repetir proc para prev dados valid
        y_prev_valid=prever(X_vk,kde_0,kde_1,priori_0,priori_1) #prev class valid set
        t_val=1-accuracy_score(Y_vk, y_prev_valid) #calc err para valid set
        train_error_nv+=t_e
        valid_error_nv+=t_val
    NB_kde_te.append(train_error_nv/5)
    NB_kde_ve.append(valid_error_nv/5)
 
#print('erro treino', NB_kde_te) 
#print('erro_validação',NB_kde_ve)    
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
y_prev_test_kde=prever(X_test,kde_0,kde_1,priori_0,priori_1) #prever com classes do X test com os kde 
test_score=1-accuracy_score(Y_test, y_prev_test_kde) #verificar score atraves da accuracy do sklearn
print('Test Error NB KDE: ',test_score)
##################################################################################
# Implementar Naive Bayes Gaussian do SKlearn      
gaussian_nb = GaussianNB()
gaussian_nb.fit(X_train, Y_train)
prev_GNB_test = gaussian_nb.predict(X_test)
error_train_gnb,error_test_gnb = 1-gaussian_nb.score(X_train, Y_train),1-gaussian_nb.score(X_test, Y_test)
print("Training Error GNB: ",np.round(error_train_gnb,4))
print("Test Error GNB: ",np.round(error_test_gnb,4)) 

###################################################################################

# implementacao do SVM com ajuste do gamma (1/sigma)

#kf = StratifiedKFold(n_splits = 5) #vem da implementacao do kdekde
SVM_te = [] #erro treino
SVM_ve = [] #erro validacao
Gamma=np.arange(0.02,0.6,0.02) #definir array para gamma 
for g in Gamma: #iterar sobrebandwith e verificar errors para cada b
    train_error_svm=0
    valid_error_svm=0 #inicializar erros a 0; será feita méia no final sum error/fold
    for train_id,valid_id in kf.split(Y_train,Y_train): # iterar sobre 5 vezes sobre as diferentes divisoes
        #fit de model NB KDE e calcula o 1-score associado (fracao classificaçoes incorrectas)  
        X_tk,Y_tk,X_vk,Y_vk=X_train[train_id],Y_train[train_id],X_train[valid_id],Y_train[valid_id]
        svm_cl=SVC(C=1,gamma=g)#default = rbf 
        svm_cl.fit(X_tk,Y_tk) #fit model
        #verificar eficacia do model nos dados treino e validacao
        t_e,t_v=1-svm_cl.score(X_tk,Y_tk), 1-svm_cl.score(X_vk,Y_vk) 
        train_error_svm+=t_e 
        valid_error_svm+=t_v
    SVM_te.append(train_error_svm)
    SVM_ve.append(valid_error_svm)


# fazer plot do gamma (eixo x) e erros no y para verificar evolucao erros com h
plt.figure()
plt.title('SVM RBF - Training vs Validation Error (gamma)')
plt.plot(Gamma,SVM_te,'blue',label='training_err') #plot erros treino acordo h
plt.plot(Gamma,SVM_ve,'red',label='validation_err') #plot erros valid acordo h
plt.xlabel('gamma value')
plt.ylabel('error')
plt.savefig('SVM.png',dpi=250) #gravar com nome requerido 
plt.close()

gamma_opt=Gamma[np.argmin(SVM_ve)]
print('gamma optimized: ',gamma_opt,' com um erro validação', min(SVM_ve))
## fit svm com gamma optimizado
SVM_test = SVC(C=1.0 , kernel = "rbf", gamma = gamma_opt)
SVM_test.fit(X_train,Y_train) 
prev_SVM_test = SVM_test.predict(X_test) #array de y_prev svm 
test_error_SVM= 1-SVM_test.score(X_test,Y_test) #erro verdadeiro 
print("SVM Test Error: ",np.round(test_error_SVM,5))

##############################################################################
# Comparação classificadores através método Macnemar e aproximate normal test
# kde - y_prev_test_kde, NB - prev_gnb_test, svm - prev_SVM_test

# Macnemar
KDE_NB,KDE_SVM,NB_SVM=macnemar(y_prev_test_kde,prev_GNB_test,prev_SVM_test,Y_test)
print(f"NB KDE vs GNB:{KDE_NB}, NV KDE vs SVM:{KDE_SVM}, GNB vs SVM: {NB_SVM}")

#stats.chi2.cdf(NB_SVM,1) = 0.65 quando p(x)>p(a) nao ha diferenca entre os classificadores
#Aproximate normal test

############################################################################################

# implementacao do SVM com ajuste do gamma (1/sigma) e c

#kf = StratifiedKFold(n_splits = 5) #vem da implementacao do kdekde
Gamma=np.arange(0.02,0.6,0.02) #definir array para gamma 
C_op=np.arange(1,60,3)
complex_SVM_te = [] #erro treino
complex_SVM_ve = [] #erro validacao
for c in C_op:
    SVM_te = [] #erro treino
    SVM_ve = [] #erro validacao
    for g in Gamma: #iterar sobrebandwith e verificar errors para cada b
        train_error_svm=0
        valid_error_svm=0 #inicializar erros a 0; será feita méia no final sum error/fold
        for train_id,valid_id in kf.split(Y_train,Y_train): # iterar sobre 5 vezes sobre as diferentes divisoes
            #fit de model NB KDE e calcula o 1-score associado (fracao classificaçoes incorrectas)  
            X_tk,Y_tk,X_vk,Y_vk=X_train[train_id],Y_train[train_id],X_train[valid_id],Y_train[valid_id]
            svm_cl=SVC(C=c,gamma=g)#default = rbf 
            svm_cl.fit(X_tk,Y_tk) #fit model
            #verificar eficacia do model nos dados treino e validacao
            t_e,t_v=1-svm_cl.score(X_tk,Y_tk), 1-svm_cl.score(X_vk,Y_vk) 
            train_error_svm+=t_e 
            valid_error_svm+=t_v
        SVM_te.append(train_error_svm)
        SVM_ve.append(valid_error_svm)
    complex_SVM_te.append(SVM_te)
    complex_SVM_ve.append(SVM_te)
    
# verificar parametros que minimiza o erro de validação
SVM_ve_op=np.array(complex_SVM_ve) #tranformar em array np
min_err_val=np.amin(SVM_ve_op)
print('erro validacao minimo: :',min_err_val)
indices=np.where(SVM_ve_op==min_err_val) #retorna indices do valor minimo m=c n =gamma
C_opt=C_op[indices[0]]
Gamma_opt=Gamma[indices[1]]

print(f'os parametros que optimizam o modelos são c={C_opt} e gamma={Gamma_opt} com um erro de validação={min_err_val}')

SVM_test_op = SVC(C=C_opt , kernel = "rbf", gamma = Gamma_opt)
SVM_test_op.fit(X_train,Y_train) 
prev_SVM_op = SVM_test_op.predict(X_test) #array de y_prev svm 
error_test_op= 1-SVM_test_op.score(X_test,Y_test) #erro verdadeiro 
print("SVM Test Error: ",np.round(test_error_SVM,5))
print(f'com os parametros optimizados o erro verdadeiro é {erro_test_op}')


