# -*- coding: utf-8 -*-
"""
Created on Mon May 21 14:45:04 2021

@author: 16319
"""
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.nonparametric.smoothers_lowess import lowess
import numpy as np
import scipy.stats as stats
from pydataset import data
import numpy as np
import sklearn as sk

Auto = pd.read_csv('Auto.csv', header=0, na_values='?').dropna()
df = pd.DataFrame(Auto)
pd.plotting.scatter_matrix(df)
xbar1 = np.mean(df["cylinders"])
#print(xbar1)
ybar = np.mean(df["mpg"])
#print(ybar)
cylinders = df["cylinders"]
one_vec = np.ones(392)
cyl_bar = xbar1 * one_vec
mpg = df["mpg"]
ybar1 = ybar * one_vec
test1 =(cylinders-cyl_bar)
test2 =(mpg-ybar1)
beta1_num = sum(test1*test2)
#print(beta1_num)
test3 = (np.square(cylinders-cyl_bar))
#print(test3)
beta1_den = sum(test3)
#print(beta1_den)
beta1= beta1_num/beta1_den
print(beta1)
displacement = df["displacement"]
displacement_bar = np.mean(df["displacement"])*one_vec
beta2_num = sum((displacement - displacement_bar)*(mpg - ybar1))                                                                  
beta2_den =sum(np.square(displacement-displacement_bar))
beta2 = beta2_num/beta2_den
print(beta2)
horsepower = df["horsepower"]
horsepower_bar = np.mean(df["horsepower"])*one_vec
beta3_num = sum((horsepower - horsepower_bar)*(mpg - ybar1))
beta3_den = sum(np.square(horsepower-horsepower_bar))
beta3 = beta3_num/beta3_den
print(beta3)
weight = df["weight"]
weight_bar = np.mean(df["weight"])*one_vec
beta4_num = sum((weight - weight_bar)*(mpg - ybar1))
beta4_den = sum(np.square(weight - weight_bar))
beta4 = beta4_num/beta4_den
print(beta4)
acceleration = df["acceleration"]
acceleration_bar = np.mean(acceleration) * one_vec
beta5_num = sum((acceleration - acceleration_bar) * (mpg - ybar1))
beta5_den= sum(np.square(acceleration - acceleration_bar))
beta5 = beta5_num/beta5_den
print(beta5)
year = df["year"]
year_bar = np.mean(df["year"]) * one_vec
beta6_num = sum((year - year_bar) * (mpg - ybar1))
beta6_den = sum(np.square(year - year_bar))
beta6 = beta6_num/beta6_den
print(beta6)
origin = df["origin"]
origin_bar = np.mean(origin) * one_vec
beta7_num = sum((origin - origin_bar) *(mpg - ybar1))
beta7_den = sum(np.square(origin - origin_bar))
beta7 = beta7_num/beta7_den
print(beta7)

betas = np.array(beta1, beta2, beta3, beta4, beta5, beta6, beta7)
print(betas)


#name = df["name"]
#name_bar = np.mean(name) * one_vec
#beta8_num = sum((name - name_bar) * (mpg - ybar1))
#beta8_den = sum(np.square(name - name_bar))
#beta8 = beta8_num/beta8_den
#print(beta8)
#X = (df[[[[[[['cylinders','displacement','horsepower','weight','acceleration','year','origin']]]]]]])

df1 = (df[['cylinders','displacement','horsepower','weight','acceleration','year','origin']])
X = np.matrix(df1)
Xt = np.matrix.transpose(X)

XtX = np.matmul(Xt,X)

XtX_inv= np.linalg.inv(XtX)
XXtX_inv = np.matmul(X,XtX_inv)

H = np.matmul(XXtX_inv,Xt)

y_vec = (df['mpg'])

yhat = np.matmul(H,y_vec)
print(yhat)
res = y_vec - yhat
print(res)
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess as  sm_lowess
top3 = abs(res).sort_values(ascending = False)[:3]
smoothed = sm_lowess(res,yhat)
plt.rcParams.update({'font.size': 16})
plt.rcParams["figure.figsize"] = (8,7)
fig, ax = plt.subplots()
ax.scatter(yhat, res, edgecolors = 'k', facecolors = 'none')
ax.plot(smoothed[:,0],smoothed[:,1],color = 'r')
ax.set_ylabel('Residuals')
ax.set_xlabel('Fitted Values')
ax.set_title('Residuals vs. Fitted')
ax.plot([min(yhat),max(yhat)],[0,0],color = 'k',linestyle = ':', alpha = .3)

for i in top3.index:
    ax.annotate(i,xy=(yhat[i],res[i]))

plt.show()

mse = sum(np.square(mpg - yhat))/392
print(mse)
rmse = np.sqrt(mse)
print(rmse)
from sklearn import linear_model
import statsmodels.formula.api as smf
#regression = sk.linear_model.LinearRegression('mpg ~ cylinders + displacement + horsepower + weight + acceleration + year + origin ', df)
#predictions = regression.predict(df)
#print(regression)
mod = smf.ols(formula = 'mpg ~ cylinders + displacement + horsepower + weight + acceleration + year + origin ', data = df)

reg = mod.fit()

print(reg.summary())
