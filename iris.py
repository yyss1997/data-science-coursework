# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
df=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',header=None)
NumDataPerClass =50
f=df.iloc[0:100,4].values
f=np.where(f=='Iris-setosa',1,-1)
Y=df.iloc[0:100,[1,3]].values
rIndex=np.random.permutation(2*NumDataPerClass)
Yr=Y[rIndex,]
fr = f[rIndex,]

Y_train= Yr[0:NumDataPerClass]
f_train= fr[0:NumDataPerClass]
Y_test= Yr[NumDataPerClass:2*NumDataPerClass]
f_test= fr[NumDataPerClass:2*NumDataPerClass]
print(Y_train.shape,f_train.shape,Y_test.shape,f_test.shape)
Ntrain=NumDataPerClass;
Ntest=NumDataPerClass;

def PercentCorrect(Inputs,targets, weights):
    targets=np.array(targets)
    Inputs=np.array(Inputs)
    N=len(targets)
    nCorrect= 0
    for n in range(N):
        OneInput = Inputs[n,:]
        if(targets[n]*np.dot(OneInput,weights)>0):
            nCorrect +=1
    return 100*nCorrect/N

a= np.random.randn(2)

print(a)

print('Initial Percentage Correct: ',PercentCorrect(Y_train,f_train, a))

MaxIter=100
alpha=0.01

P_train = np.zeros(MaxIter)
P_test = np.zeros(MaxIter)

for iter in range(MaxIter):
    r=np.floor(np.random.rand()*Ntrain).astype(int)
    y=Y_train[r,:]
    
    if(f_train[r]*np.dot(y,a)<0):
        a+=alpha*f_train[r]*y
    
    P_train[iter] = PercentCorrect(Y_train, f_train,a);
    P_test[iter]=PercentCorrect(Y_test, f_test, a);
    
print('Percenttage Correct After Trainin: ',
    PercentCorrect(Y_train,f_train, a),
    PercentCorrect(Y_test,f_test, a))
    
plt.plot(range(MaxIter),P_train, 'b', range(MaxIter),P_test, 'r')
plt.grid(True)
plt.gca().legend(('Training Set','Test Set'))