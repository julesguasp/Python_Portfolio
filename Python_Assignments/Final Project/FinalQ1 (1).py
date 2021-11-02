# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 16:17:22 2021

@author: 16319
"""

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import graphviz

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import confusion_matrix, mean_squared_error, classification_report
from sklearn.preprocessing import scale

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00464/superconduct.zip"
data = pd.read_csv('train.csv')
data = data.dropna()
data.astype('int64').dtypes
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
X = scale(data.iloc[:,0:81])

y = scale(data.iloc[:,81].astype('int64'))

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25, random_state=0)
from sklearn.preprocessing import LabelEncoder
lab_enc = LabelEncoder()
y_train = lab_enc.fit_transform(y_train)
#1.  Use regression trees to construct the predictive model.
from sklearn.model_selection import GridSearchCV

max_depth_range = np.arange(1,11,1)
mss_range = np.arange(2,11,1)

tuned_parameters_lr = [{'max_depth': max_depth_range,
                        'min_samples_split': mss_range, }]
lr = GridSearchCV(DecisionTreeRegressor(), tuned_parameters_lr, cv=KFold(n_splits=10), scoring = 'neg_mean_squared_error')
lr.fit(X_train.astype('int64'), y_train)
print(lr.best_params_)
#max_depth : 10
#mss: 6

linear = DecisionTreeRegressor(max_depth = 10, min_samples_split=6)
linear.fit(X_train.astype('int64'),y_train)
y_pred_lr_train = linear.predict(X_train)

from sklearn.metrics import mean_squared_error
print("MSE train:", mean_squared_error(y_train,y_pred_lr_train))
y_pred_lr_test = linear.predict(X_test)
print("MSE test:", mean_squared_error(y_test,y_pred_lr_test))


RT_trainmse = mean_squared_error(y_train,y_pred_lr_train)
RT_testmse = mean_squared_error(y_test,y_pred_lr_test)      
#2.  Use bagging classification to construct your classifier.  Report also thebagging important variables.
from sklearn.ensemble import BaggingRegressor

n_est_range = np.arange(10,21,1)
max_samples_range = np.arange(1,11,1)
max_features_range = np.arange(1,11,1)

tuned_parameters_lrbag = [{ 'n_estimators': n_est_range,
                            'max_features': max_features_range,
                            'max_samples_range': max_samples_range
    }]

lr_bag = GridSearchCV(BaggingRegressor(), tuned_parameters_lrbag, cv=KFold(n_splits=10), scoring = 'neg_mean_squared_error')
lr_bag.fit(X_train.astype('int64'),y_train)
print(lr_bag.best_params_)
#{'max_features': 9, 'max_samples': 10, 'n_estimators': 18}

linear_bag = BaggingRegressor(max_features = 9, max_samples = 10, n_estimators = 18)
linear_bag.fit(X_train,y_train)
y_pred_lr_bag_train = linear_bag.predict(X_train)
print("MSE train:", mean_squared_error(y_train,y_pred_lr_bag_train))
y_pred_lr_bag_test = linear_bag.predict(X_test)
print("MSE test:", mean_squared_error(y_test,y_pred_lr_bag_test))

LB_trainmse = mean_squared_error(y_train,y_pred_lr_bag_train)
LB_testmse = mean_squared_error(y_test,y_pred_lr_bag_test)


Importance_linear_bag = pd.DataFrame({'Importance': linear.feature_importances_*100})
	
Importance_linear_bag.sort_values(by = 'Importance',
	axis = 0,
	ascending = True). plot(kind = 'barh',
		color = 'r', )

plt.xlabel('Variable Importance')
plt.gca().legend_ = None


n_est_rf_range = np.arange(10,100,10)

tuned_parameters_lrrf = [{'max_features': max_features_range,
                          'n_estimators': n_est_rf_range
                          }]
lr_rf = GridSearchCV(RandomForestRegressor(), tuned_parameters_lrrf, cv=KFold(n_splits=3), scoring = 'neg_mean_squared_error')
lr_rf.fit(X_train.astype('int64'),y_train)
print(lr_rf.best_params_)
#{'max_features': 4, 'n_estimators': 90}

linear_rf = RandomForestRegressor(max_features = 4, n_estimators = 90)
linear_rf.fit(X_train,y_train)
y_pred_lr_rf_train = linear_rf.predict(X_train)
print("MSE train:", mean_squared_error(y_train,y_pred_lr_rf_train))

y_pred_lr_rf_test = lr_rf.predict(X_test)
print("MSE test:", mean_squared_error(y_test,y_pred_lr_rf_test))

RF_trainmse = mean_squared_error(y_train,y_pred_lr_rf_train)
RF_testmse = mean_squared_error(y_test,y_pred_lr_rf_test)

Importance_linear_rf = pd.DataFrame({'Importance': linear_rf.feature_importances_*100})
	
Importance_linear_rf.sort_values(by = 'Importance',
	axis = 0,
	ascending = True). plot(kind = 'barh',
		color = 'r', )

plt.xlabel('Variable Importance')
plt.gca().legend_ = None

###
learning_rate_range = np.arange(0.1,1,0.1)

tuned_parameters_lrboost = [{ 'n_estimators': n_est_range,
                            'learning_rate': learning_rate_range
    }]

lr_boost = GridSearchCV(GradientBoostingRegressor(), tuned_parameters_lrboost, cv=KFold(n_splits=3), scoring = 'neg_mean_squared_error')
lr_boost.fit(X_train.astype('int64'),y_train)
print(lr_boost.best_params_)
#{'learning_rate': 0.7000000000000001, 'n_estimators': 20}

linear_boost = GradientBoostingRegressor(learning_rate = 0.7, n_estimators = 20)
linear_boost.fit(X_train,y_train)
y_pred_lr_boost_train = linear_boost.predict(X_train)
print("MSE train:", mean_squared_error(y_train,y_pred_lr_boost_train))

y_pred_lr_boost_test = linear_boost.predict(X_test)
print("MSE test:", mean_squared_error(y_test,y_pred_lr_boost_test))

Boost_trainmse = mean_squared_error(y_train,y_pred_lr_boost_train)
Boost_testmse = mean_squared_error(y_test,y_pred_lr_boost_test)

Importance_linear_boost = pd.DataFrame({'Importance': linear_boost.feature_importances_*100})
	
Importance_linear_boost.sort_values(by = 'Importance',
	axis = 0,
	ascending = True). plot(kind = 'barh',
		color = 'r', )

plt.xlabel('Variable Importance')
plt.gca().legend_ = None

linreg = LinearRegression()
linreg.fit(X_train,y_train)
y_pred_linreg_train = linreg.predict(X_train)
print("MSE train:", mean_squared_error(y_train,y_pred_linreg_train))

y_pred_linreg_test = linreg.predict(X_test)
print("MSE test:", mean_squared_error(y_test,y_pred_linreg_test))

lr_trainmse = mean_squared_error(y_train,y_pred_linreg_train)
lr_testmse = mean_squared_error(y_test,y_pred_linreg_test)

#Comparing MSE Results

RT_mse_avg = (RT_trainmse + RT_testmse)/2
LB_mse_avg = (LB_trainmse + LB_testmse)/2
RF_mse_avg = (RF_trainmse + RF_testmse)/2
Boost_mse_avg = (Boost_trainmse + Boost_testmse)/2
lr_mse_avg = (lr_trainmse + lr_testmse)/2
print("Regression Tree MSE Avg: ", RT_mse_avg)
print("Bagging MSE Avg", LB_mse_avg)
print("Random Forest MSE Avg", RF_mse_avg)
print("Boosting MSE Avg", Boost_mse_avg)
print("Linear Regression MSE Avg", lr_mse_avg)


