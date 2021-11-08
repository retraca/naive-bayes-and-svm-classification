# -*- coding: utf-8 -*-
import numpy as np
   
def McNemar(predNB,predGNB,predSVM,Y_test):
    
    bool_NB = predNB == Y_test
    bool_NB = np.reshape(bool_NB,(np.size(bool_NB),len(bool_NB)))
    bool_GNB = predGNB == Y_test
    bool_SVM = predSVM == Y_test
    
    e01_NBGNB = 0
    e10_NBGNB = 0
    e01_NBSVM = 0
    e10_NBSVM = 0
    e01_GNBSVM = 0
    e10_GNBSVM = 0
    
    for i in range(0,len(bool_NB)):
        if bool_NB[i] == True and bool_GNB[i] == False:
            e10_NBGNB += 1
        
        elif bool_NB[i] == False and bool_GNB[i] == True:
            e01_NBGNB += 1
    
    for i in range(0,len(bool_SVM)):
        if bool_NB[i] == True and bool_SVM[i] == False:
            e10_NBSVM += 1
        
        elif bool_NB[i] == False and bool_SVM[i] == True:
            e01_NBSVM += 1
            
    for i in range(0,len(bool_GNB)):
        if bool_GNB[i] == True and bool_SVM[i] == False:
            e10_GNBSVM += 1
        
        elif bool_GNB[i] == False and bool_SVM[i] == True:
            e01_GNBSVM += 1
     
    NB_vs_GNB = ((np.abs(e01_NBGNB-e10_NBGNB)-1)**2)/(e10_NBGNB+e01_NBGNB)
    NB_vs_SVM = ((np.abs(e01_NBSVM-e10_NBSVM)-1)**2)/(e10_NBSVM+e01_NBSVM)    
    GNB_vs_SVM = ((np.abs(e01_GNBSVM-e10_GNBSVM)-1)**2)/(e10_GNBSVM+e01_GNBSVM)
    
    return NB_vs_GNB,NB_vs_SVM,GNB_vs_SVM
