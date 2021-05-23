# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 14:24:14 2021

@author: Carter Hogan
"""
# import necessary packages 
import matplotlib.pyplot as plt
import seaborn; seaborn.set()

import math

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

from numpy.random import default_rng

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

from scipy.stats import pearsonr as pcorr

#ipython magic function, helps display of plots in a notebook
#%matplotlib inline
# read in the data set off of the 
cd = pd.read_csv("http://www.rob-mcculloch.org/data/susedcars.csv")
cd = cd[['price','mileage','year']]
cd['price'] = cd['price']/1000
cd['mileage'] = cd['mileage']/1000
print(cd.head()) # head just prints out the first few rows

n = cd.shape[0]
pin = .75 # percent of data to put in train
rng = np.random.default_rng(seed=42)
ii = rng.choice(range(n),size=int(pin*n),replace=False)
indtr = np.zeros(n,dtype=bool)
indtr[ii] = True

cdtrain = cd[indtr]
cdtest = cd[~indtr]
print(cdtrain.shape)
print(cdtest.shape)

X = cd.iloc[:,[1,2]].to_numpy()
y = cd['price'].to_numpy()
print(X.shape)
print(y.shape)

#use sklearn train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X,y, random_state=99,test_size=.25)
print(Xtrain.shape,ytrain.shape)
print(Xtest.shape,ytest.shape)

lmmod = LinearRegression(fit_intercept=True)
lmmod.fit(Xtrain,ytrain)

yhtest = lmmod.predict(Xtest)
plt1 = plt.scatter(yhtest,ytest)
plt.xlabel('yhtest'); plt.ylabel('y')

def rmse(y,yh):
    return(math.sqrt(np.mean((y-yh)**2)))

def mabe(y,yh):
    return(np.mean(np.abs(y-yh)))

print('rmse: ',rmse(ytest,yhtest))
print('mabe: ',mabe(ytest,yhtest))
print('R-squared: ',pcorr(ytest,yhtest)[0]**2)


## Part 1 does logging y help? First with both x's
logp = np.log(cd['price'])
Xtrain, Xtest, ltrain, ltest = train_test_split(X,logp, random_state=99,test_size=.25)
lmmod1= LinearRegression(fit_intercept=True)
lmmod1.fit(Xtrain,ltrain)

lhtest = lmmod1.predict(Xtest)
plt2 = plt.scatter(lhtest,ltest)
plt.xlabel('lhtest'); plt.ylabel('logy')


print('rmse: ',rmse(ltest,lhtest))
print('mabe: ',mabe(ltest,lhtest))
print('R-squared: ',pcorr(ltest,lhtest)[0]**2)

## Now without year 
mile =  cd.iloc[:,[1,]].to_numpy()
mtrain, mtest, ltrain, ltest = train_test_split(mile,logp, random_state=99,test_size=.25)

lmmod2= LinearRegression(fit_intercept=True)
lmmod2.fit(mtrain,ltrain)
lh2test = lmmod2.predict(mtest)
plt3 = plt.scatter(lh2test,ltest)
plt.xlabel('lh2test'); plt.ylabel('logy')

print('rmse: ',rmse(ltest,lh2test))
print('mabe: ',mabe(ltest,lh2test))
print('R-squared: ',pcorr(ltest,lh2test)[0]**2)

fig, (plt1,plt2,plt3) = plt.subplots(nrows=3, ncols=1)

# Here we can already see that both variables describe a greater part of the variation of price in combination

## Part 2 Does color help?

cd1 = pd.read_csv("http://www.rob-mcculloch.org/data/susedcars.csv")
cd1.columns.values
cd1 = cd1.iloc[:,[3,4,5,0]]
cd1['price'] = cd1['price']/1000
cd1['mileage'] = cd1['mileage']/1000
print(cd1.head())

one_hot = LabelBinarizer()
cdums = one_hot.fit_transform(cd1['color'])
print(type(cdums))
print(cdums.shape)
cdums[0:10,:]

cd1['color'][0:10]

X1 = np.hstack([cd1.iloc[:,0:1].to_numpy(),cdums[:,1:4]])
X1[0:5,:]

lmmod3 = LinearRegression(fit_intercept=True)
lmmod3.fit(X1,logp)
print(lmmod3.intercept_,lmmod3.coef_)
