"""
This code implements the computation of some particular fairness metrics for tree
trained with sklearn

These metrics are introduced in :
    Zafar, Muhammad Bilal, et al. "Fairness Constraints: A Flexible Approach for Fair Classification."
    Journal of Machine Learning Research 20.75 (2019): 1-42.
    
Last modification : 10 November 2020
           Author : Delchevalerie Valentin
            email : valentin.delchevalerie@unamur.be
"""


import numpy as np


# Disparate impact
def getDisparateImpact(X_train, algebraicTree, controled_axis=-1):
    """
    This function returns the Disparate Impact (DI) of the tree.
    
    A DI close to zero means that the model is fair.
    """
    
    N = np.shape(X_train)[0]
    z_bar = np.mean(X_train[:,controled_axis])
    
    DI = 0.0
    for i in range(N):
        DI += (X_train[i,controled_axis] - z_bar) * algebraicTree.signedDistance(X_train[i,:])

    return np.abs(DI/N)


# Disparate mistreatment w.r.t overall misclassification
def getDisparateMistreatment_overall(X_train, y_train, algebraicTree, controled_axis=-1):
    """
    This function returns the Disparate Mistreatment (DM) of the tree
    w.r.t overall misclassification.
    
    A DM close to zero means that the model is fair.
    """
    
    N = np.shape(X_train)[0]
    N_1 = np.shape(np.where(X_train[:,controled_axis] == 1)[0])[0]
    N_0 = N - N_1
    coeff1 = N_1/N
    coeff2 = N_0/N 
    
    DM = 0.0
    for i in range(N):
        if X_train[i,controled_axis] == 0:
            DM -= coeff1 * algebraicTree.misclassifiedDistance(X_train[i,:], y_train[i])
        else:
            DM += coeff2 * algebraicTree.misclassifiedDistance(X_train[i,:], y_train[i])
            
    return np.abs(DM)


# Disparate mistreatment w.r.t false positive rates and false negative rates
def getDisparateMistreatment_fpfn(X_train, y_train, algebraicTree, controled_axis=-1):
    """
    This function returns the Disparate Mistreatment (DM) of the tree
    w.r.t false positive rates and false negative rates.
    
    A DM close to zero means that the model is fair.
    """
   
    indices = np.where((y_train == 0) | (y_train == -1))[0]
    N_neg = np.shape(indices)[0]
    N_1_neg = np.shape(np.where((X_train[:,controled_axis] == 1) & ((y_train == 0) | (y_train == -1)))[0])[0]
    N_0_neg = np.shape(np.where((X_train[:,controled_axis] == 0) & ((y_train == 0) | (y_train == -1)))[0])[0]
    coeff1 = N_1_neg/N_neg
    coeff2 = N_0_neg/N_neg
    
    DM = 0.0
    for i in indices:
        if X_train[i,controled_axis] == 0:
            DM -= coeff1 * algebraicTree.misclassifiedDistance(X_train[i,:], y_train[i])
        else:
            DM += coeff2 * algebraicTree.misclassifiedDistance(X_train[i,:], y_train[i])
            
    return np.abs(DM)