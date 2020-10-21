#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 19:19:07 2020

@author: carterhogan
@title: Time Series Analysis on U.S. Historic Economic Data 1875-1983
"""
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
# Download the Time Series from the National Bureau of Economic Research
df = pd.read_csv(filepath_or_buffer = 'http://data.nber.org/data/abc/abcq.csv')

print(df.head(10))
print('\n Data Types:')
print (df.dtypes)
df = df.sort_values(by =['year'], ascending = True)
df.head(10)
## Gross Overview of Data from 1875 on GNP, Wholesale prices and the Money Supply

#GNP 
boxplotGNP = df.boxplot(column = ['GNP','RGNP72'])
df.plot(kind = 'line', x= 'year',y = ['GNP', 'RGNP72', 'TRGNP'])
plt.show()
# End the Fed?
boxplotMoneySupply = df.boxplot(column = ['M1','M2','BASE'])
df.plot(kind = 'line', x = 'year',y=['M1','M2','BASE'])
plt.show()
#A glimpse at prices 
boxplotprices = df.boxplot(column = ['WPRICE67', 'CSTOCK'])
df.plot(kind = 'line', x= 'year', y= ['WPRICE67','CSTOCK'])
## The data set has data for all variables starting at 1919 so we restrict the data set to those years 
dfsub = df.iloc[176:435,]
#Since the data has set these variables in real terms relative to the value of a $ in 1972, we use those instead of nominal variables 
#consumption of Durables, Nondurables & Services, and Investment 
ax =dfsub.plot(kind = 'line', x = 'year', y = ['CDUR72','CNDUR72','IRES72'])
ax
#Imports and Exports 
bx =dfsub.plot(kind='line', x= 'year', y= ['MPT72','XPT72'])
bx
## Creating a Model for GNP based off of Consumption, Investment, Exports - Imports, and Government expenditures (which is the GDP formula but GNP and GDP are highly correlated in that GNP = GDP + Net Income)
# Check GNP distribution - could it be less skewed under a log function?
print('Skewness of RGNP72 is', dfsub['RGNP72'].skew())
sns.distplot(dfsub['RGNP72'])
lRGNP72 = np.log(dfsub['RGNP72'])
sns.distplot(lRGNP72)
print('Skewness of the log transformation of RGNP72 is', lRGNP72.skew())
# This looks much more like a normal distribution which takes care of our dependent variable, let's check out the other variables 
print('Skewness of Consumption of Durable Goods is', dfsub['CDUR72'].skew())
sns.distplot(dfsub['CDUR72'])
lCDUR72 = np.log(dfsub['CDUR72'])
sns.distplot(lCDUR72)
print('Skewness of log transformation of CDUR72 is', lCDUR72.skew())
# Consumption of Nondurables & Services
print('Skewness of Consumption of Nondurable Goods & Services is', dfsub['CNDUR72'].skew())
sns.distplot(dfsub['CNDUR72'])
lCNDUR72 = np.log(dfsub['CNDUR72'])
sns.distplot(lCNDUR72)
print('Skewness of log transformation of CNDUR72 is', lCNDUR72.skew())
#Since the change of them is really what we're focused on for a multivariable linear regression we will transform all to logs accordingly except trade because it takes negative values
lIRES72 = np.log(dfsub['IRES72'])
trade = dfsub['XPT72']-dfsub['MPT72']
lGOVPUR72 = np.log(dfsub['GOVPUR72'])
# Do OLS Regression 

