# -*- coding: utf-8 -*-
"""
Created on Tue May 22 13:47:33 2021

@author: 16319
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import datasets

data_set = datasets.load_breast_cancer()

X=data_set.data
y=data_set.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

K1 = pd.DataFrame(X_test[y_test == 0])
K2 = pd.DataFrame(X_test[y_test == 1])
#X_train_df = pd.DataFrame(X_train)
#X_test_df = pd.DataFrame(X_test)
#y_train_df = pd.DataFrame(y_train)
#y_test_df = pd.DataFrame(y_test)



xb1= np.mean(K1)
xb2 = np.mean(K2)

x1t = (K1.T)
x2t = (K2.T)

s1 = np.cov(x1t)
s2 = np.cov(x2t)

p1=p2=0.5

n1=n2=2
n = n1+n2
k=2

sp = (((n1-1)*s1 +(n2-1)*s2)/(n-k))

from numpy.linalg import inv
spi = inv(sp)

xnew = X_test[0, :]

t1 = np.matmul(xb1.T, spi)
cst1 = np.matmul(t1, xb1)
e1=np.matmul(t1, xnew)

det1= e1-(1/2)*cst1+np.log(p1)
print(det1)

t2 = np.matmul(xb2.T, spi)
cst2 = np.matmul(t2, xb2)
e2=np.matmul(t2, xnew)

det2= e2-(1/2)*cst2+np.log(p2)
print(det2)

print(np.max([det1,det2]))
#classify xnew as 0 

def lda_delta(xnew, ynew): 
    x = 0
    while x < len(xnew):
            x0 = xnew[x, :]
            K1 = pd.DataFrame(xnew[ynew == 0])
            K2 = pd.DataFrame(xnew[ynew == 1])
            xb1= np.mean(K1)
            xb2 = np.mean(K2)
            
            x1t = (K1.T)
            x2t = (K2.T)
            
            s1 = np.cov(x1t)
            s2 = np.cov(x2t)
            
            p1=p2=0.5
            
            n1=n2=2
            n = n1+n2
            k=2
            
            sp = (((n1-1)*s1 +(n2-1)*s2)/(n-k))
            
            from numpy.linalg import inv
            spi = inv(sp)
            t1 = np.matmul(xb1.T, spi)
            cst1 = np.matmul(t1, xb1)
            e1=np.matmul(t1, x0)
            
            det0= e1-(1/2)*cst1+np.log(p1)
           # print(det0)
            
            t2 = np.matmul(xb2.T, spi)
            cst2 = np.matmul(t2, xb2)
            e2=np.matmul(t2, x0)
            
            det1= e2-(1/2)*cst2+np.log(p2)
            #print(det1)
            
            if np.max([det0,det1]) == det0:
                print(0)
            elif np.max([det0,det1]) == det1:
                print(1)
            else:
                print("error")
            x += 1 
    

n = np.matrix(lda_delta(X_train, y_train))

n.to_excel('./confusion0.xlsx', sheet_name = 'train_results', index=False)
y_train_mat = pd.DataFrame(y_train)
y_train_mat.to_excel('./confusion1.xlsx', sheet_name = 'train', index=False)

z = np.matrix(lda_delta(X_test, y_test))
y_test_mat = pd.DataFrame(y_test)

z.to_excel('./confusion2.xlsx', sheet_name = 'test_results')
y_test_mat.to_excel('./confusion3.xlsx', sheet_name = 'test', index=False)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import neighbors

lda = LinearDiscriminantAnalysis()
pred = lda.fit(X_train, y_train).predict(X_test)

lda.priors_

lda.means_

lda.coef_

confusion_matrix(y_test, pred).T

print(classification_report(y_test, pred, digits=3))

pred_p = lda.predict_proba(X_test)

a1=np.unique(pred_p[:,1]>0.5, return_counts=True)
a1
np.unique(pred_p[:,1]>0.9, return_counts=True)

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


