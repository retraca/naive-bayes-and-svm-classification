# -*- coding: utf-8 -*-
from sklearn.neighbors.kde import KernelDensity
import numpy as np
import math

class NaiveBayes:
    
    def __init__(self,bw,feats):
        
        self.feats = feats
        self.bw = bw
    
    #fit the training sets of cross validation
    def fit(self,x_train,y_train):
        
        self.priory1_train = len(x_train[y_train==1])/len(x_train)
        self.priory0_train = len(x_train[y_train==0])/len(x_train)
        self.kdetrain1 = dict()
        self.kdetrain0 = dict()
        xtrain1 = x_train[y_train==1,:]
        xtrain0 = x_train[y_train==0,:]
        ytrain1 = y_train[y_train==1]
        ytrain0 = y_train[y_train==0]
        
        for feat in self.feats:
            
            kde_train0 = KernelDensity(bandwidth=self.bw, kernel='gaussian')
            kde_train0.fit(xtrain0[:,[feat-1]],ytrain0[feat-1])
            self.kdetrain0[feat] = (kde_train0)
            
            kde_train1 = KernelDensity(bandwidth=self.bw, kernel='gaussian')
            kde_train1.fit(xtrain1[:,[feat-1]],ytrain1[feat-1])
            self.kdetrain1[feat] = (kde_train1)
        
        return 0
    
    #predict after fitting with the training set
    def predict(self,x,y):
        
        self.Cpred_0 = []
        self.Cpred_1 = []
        preds_t0 = []
        preds_t1 = []

        for i in self.kdetrain0.keys():
            
            pred_t0 = self.kdetrain0[i].score_samples(x[:,[i-1]])
            pred_t1 = self.kdetrain1[i].score_samples(x[:,[i-1]])
            preds_t0.append(pred_t0)
            preds_t1.append(pred_t1)

        summedpreds_t0 = np.sum(preds_t0,axis=0)
        summedpreds_t1 = np.sum(preds_t1,axis=0)

        for i in range(0,len(summedpreds_t0)):
            
            Cpred0 = math.log(self.priory0_train) + summedpreds_t0[i]
            Cpred1 = math.log(self.priory1_train) + summedpreds_t1[i]
            self.Cpred_0.append(Cpred0)
            self.Cpred_1.append(Cpred1)
        
        a = np.reshape(self.Cpred_0,(1,len(self.Cpred_0)))
        b = np.reshape(self.Cpred_1,(1,len(self.Cpred_1)))       
        self.k = (a <= b).astype(int)  
        
        return self.k
    
    #scores the predictons and returns a error
    def score(self,y):
        
        y = np.reshape(y,(1,len(y))) 
        num_false = 0
        num_true = 0
        
        for i in range(0,len(y)):
            
            for j in range(0,np.size(y)):
                
                if self.k[i][j] != y[i][j]:
                    
                    num_false += 1
                
                elif self.k[i][j] == y[i][j]:
                    
                    num_true += 1
                    
        error = num_false/np.size(y)
        
        return error

        
        
        
        
        
        
        

