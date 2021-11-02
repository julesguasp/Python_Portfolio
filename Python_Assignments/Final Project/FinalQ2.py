# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 16:47:01 2021

@author: 16319
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split

from sklearn.metrics import auc, roc_curve
from sklearn import datasets
from sklearn.metrics import confusion_matrix, classification_report, precision_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn import metrics
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold


data_set = datasets.load_breast_cancer()
X= preprocessing.scale(data_set.data)
y= data_set.target
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25, random_state=0)

#1.  Use classification trees to construct your classifier.

max_depth_range = np.arange(1,11,1)
mss_range = np.arange(2,11,1)

tuned_parameters_lr = [{'max_depth': max_depth_range,
                        'min_samples_split': mss_range, }]
lr = GridSearchCV(DecisionTreeClassifier(), tuned_parameters_lr, cv=KFold(n_splits=3), scoring = 'accuracy')
lr.fit(X_train.astype('int64'), y_train)
print(lr.best_params_)
#{'max_depth': 9, 'min_samples_split': 5}

linear = DecisionTreeClassifier(max_depth = 9, min_samples_split=5)
linear.fit(X_train.astype('int64'),y_train)
y_pred_lr_train = linear.predict(X_train)
ct_mer_train = 1 - metrics.accuracy_score(y_train,y_pred_lr_train)
conf__lr = pd.DataFrame(metrics.confusion_matrix(y_train, y_pred_lr_train))
TP_lr = conf__lr.iloc[0,0]
FP_lr = conf__lr.iloc[1,0]
FN_lr = conf__lr.iloc[0,1]
TN_lr = conf__lr.iloc[1,1]
print("Misclassification Error Rate", ct_mer_train)
print("Training Confusion Matrix \n", metrics.confusion_matrix(y_train, y_pred_lr_train))
print("Recall:", TP_lr/(TP_lr+FN_lr))
print("Specificity:", TN_lr/(TN_lr+FP_lr))
print("Fallout:", FP_lr/(FP_lr + TN_lr))
print("PPV:", TP_lr/(TP_lr + FP_lr))
print("Accuracy:", (TP_lr + TN_lr)/(TP_lr +TN_lr + FP_lr + FN_lr))

y_pred_lr_test = linear.predict(X_test)
ct_mer_test = 1 - metrics.accuracy_score(y_test,y_pred_lr_test)

print("Misclassification Error Rate", ct_mer_test)
print("Testing Confusion Matrix \n", metrics.confusion_matrix(y_test, y_pred_lr_test))

ct_mer_test = 1 - metrics.accuracy_score(y_test,y_pred_lr_test)
conf__lr1 = pd.DataFrame(metrics.confusion_matrix(y_test, y_pred_lr_test))

TP_lr1 = conf__lr1.iloc[0,0]
FP_lr1 = conf__lr1.iloc[1,0]
FN_lr1 = conf__lr1.iloc[0,1]
TN_lr1 = conf__lr1.iloc[1,1]
print("Misclassification Error Rate", ct_mer_test)
print("Recall:", TP_lr1/(TP_lr1+FN_lr1))
print("Specificity:", TN_lr1/(TN_lr1+FP_lr1))
print("Fallout:", FP_lr1/(FP_lr1 + TN_lr1))
print("PPV:", TP_lr1/(TP_lr1 + FP_lr1))
print("Accuracy:", (TP_lr1 + TN_lr1)/(TP_lr1 +TN_lr1 + FP_lr1 + FN_lr1))

#2.  Use bagging classification to construct your classifier.  Report also thebagging important variables.
from sklearn.ensemble import BaggingClassifier

n_est_range = np.arange(1,101,10)
max_sample_range = np.arange(1,10,1)
max_features_range = np.arange(1,10,1)


tuned_parameters_bag = [{ 'n_estimators': n_est_range,
                        'max_samples': max_sample_range,
                        'max_features': max_features_range
    }]
bag = GridSearchCV(BaggingClassifier(), tuned_parameters_bag, cv=KFold(n_splits=3), scoring = 'accuracy')
bag.fit(X_train.astype('int64'), y_train)
print(bag.best_params_)
#{'max_features': 7, 'max_samples': 9, 'n_estimators': 51}

bagging = BaggingClassifier(max_features = 7, max_samples = 9, n_estimators = 51)
bagging.fit(X_train.astype('int64'),y_train)
y_pred_bag_train = bagging.predict(X_train)
bag_mer_train = 1 - metrics.accuracy_score(y_train,y_pred_bag_train)
conf__bag = pd.DataFrame(metrics.confusion_matrix(y_train, y_pred_bag_train))
TP_bag = conf__bag.iloc[0,0]
FP_bag = conf__bag.iloc[1,0]
FN_bag = conf__bag.iloc[0,1]
TN_bag = conf__bag.iloc[1,1]
print("Misclassification Error Rate", bag_mer_train)
print("Training Confusion Matrix \n", metrics.confusion_matrix(y_train, y_pred_bag_train))
print("Recall:", TP_bag/(TP_bag+FN_bag))
print("Specificity:", TN_bag/(TN_bag+FP_bag))
print("Fallout:", FP_bag/(FP_bag + TN_bag))
print("PPV:", TP_bag/(TP_bag + FP_bag))
print("Accuracy:", (TP_bag + TN_bag)/(TP_bag +TN_bag + FP_bag + FN_bag))

y_pred_bag_test = bagging.predict(X_test)
bag_mer_test = 1 - metrics.accuracy_score(y_test,y_pred_bag_test)

print("Misclassification Error Rate", bag_mer_test)
print("Testing Confusion Matrix \n", metrics.confusion_matrix(y_test, y_pred_bag_test))

bag_mer_test = 1 - metrics.accuracy_score(y_test,y_pred_bag_test)
conf__bag1 = pd.DataFrame(metrics.confusion_matrix(y_test, y_pred_bag_test))

TP_bag1 = conf__bag1.iloc[0,0]
FP_bag1 = conf__bag1.iloc[1,0]
FN_bag1 = conf__bag1.iloc[0,1]
TN_bag1 = conf__bag1.iloc[1,1]
print("Misclassification Error Rate", bag_mer_test)
print("Recall:", TP_bag1/(TP_bag1+FN_bag1))
print("Specificity:", TN_bag1/(TN_bag1+FP_bag1))
print("Fallout:", FP_bag1/(FP_bag1 + TN_bag1))
print("PPV:", TP_bag1/(TP_bag1 + FP_bag1))
print("Accuracy:", (TP_bag1 + TN_bag1)/(TP_bag1 +TN_bag1 + FP_bag1 + FN_bag1))

Importance_bagging = pd.DataFrame({'Importance': linear.feature_importances_*100})
	
Importance_bagging.sort_values(by = 'Importance',
	axis = 0,
	ascending = True). plot(kind = 'barh',
		color = 'r', )

plt.xlabel('Variable Importance')
plt.gca().legend_ = None

from sklearn.ensemble import RandomForestClassifier

n_estim_range = np.arange(80,120,10)
max_depth_range = np.arange(1,10,1)
min_samples_range = np.arange(2,10,1)


tuned_parameters_rf = [{ 'n_estimators': n_estim_range,
                        'max_depth': max_depth_range,
                        'min_samples_split': min_samples_range
    }]
rf = GridSearchCV(RandomForestClassifier(), tuned_parameters_rf, cv=KFold(n_splits=3), scoring = 'accuracy')
rf.fit(X_train.astype('int64'), y_train)
print(rf.best_params_)
#{'max_depth': 8, 'min_samples_split': 3, 'n_estimators': 80}

rforest = RandomForestClassifier(max_depth = 8, min_samples_split = 3, n_estimators = 80)
rforest.fit(X_train.astype('int64'),y_train)
y_pred_rf_train = rforest.predict(X_train)
rf_mer_train = 1 - metrics.accuracy_score(y_train,y_pred_rf_train)
conf__rf = pd.DataFrame(metrics.confusion_matrix(y_train, y_pred_rf_train))
TP_rf = conf__rf.iloc[0,0]
FP_rf = conf__rf.iloc[1,0]
FN_rf = conf__rf.iloc[0,1]
TN_rf = conf__rf.iloc[1,1]
print("Misclassification Error Rate", rf_mer_train)
print("Training Confusion Matrix \n", metrics.confusion_matrix(y_train, y_pred_rf_train))
print("Recall:", TP_rf/(TP_rf+FN_rf))
print("Specificity:", TN_rf/(TN_rf+FP_rf))
print("Fallout:", FP_rf/(FP_rf + TN_rf))
print("PPV:", TP_rf/(TP_rf + FP_rf))
print("Accuracy:", (TP_rf + TN_rf)/(TP_rf +TN_rf + FP_rf + FN_rf))

y_pred_rf_test = rforest.predict(X_test)
rf_mer_test = 1 - metrics.accuracy_score(y_test,y_pred_rf_test)

print("Misclassification Error Rate", rf_mer_test)
print("Testing Confusion Matrix \n", metrics.confusion_matrix(y_test, y_pred_rf_test))

rf_mer_test = 1 - metrics.accuracy_score(y_test,y_pred_rf_test)
conf__rf1 = pd.DataFrame(metrics.confusion_matrix(y_test, y_pred_rf_test))

TP_rf1 = conf__rf1.iloc[0,0]
FP_rf1 = conf__rf1.iloc[1,0]
FN_rf1 = conf__rf1.iloc[0,1]
TN_rf1 = conf__rf1.iloc[1,1]
print("Misclassification Error Rate", rf_mer_test)
print("Recall:", TP_rf1/(TP_rf1+FN_rf1))
print("Specificity:", TN_rf1/(TN_rf1+FP_rf1))
print("Fallout:", FP_rf1/(FP_rf1 + TN_rf1))
print("PPV:", TP_rf1/(TP_rf1 + FP_rf1))
print("Accuracy:", (TP_rf1 + TN_rf1)/(TP_rf1 +TN_rf1 + FP_rf1 + FN_rf1))

Importance_rforest = pd.DataFrame({'Importance': rforest.feature_importances_*100})
	
Importance_rforest.sort_values(by = 'Importance',
	axis = 0,
	ascending = True). plot(kind = 'barh',
		color = 'r', )

plt.xlabel('Variable Importance')
plt.gca().legend_ = None

from sklearn.ensemble import GradientBoostingClassifier

learn_rate_range = np.arange(0.1,1,.1)
subsample_range = np.arange(0.1,1,.1)


tuned_parameters_boost = [{ 'learning_rate': learn_rate_range,
                           'n_estimators': n_estim_range,
                           'subsample': subsample_range
    }]
boost = GridSearchCV(GradientBoostingClassifier(), tuned_parameters_boost, cv=KFold(n_splits=3), scoring = 'accuracy')
boost.fit(X_train.astype('int64'), y_train)
print(boost.best_params_)
#{'max_depth': 8, 'min_samples_split': 3, 'n_estimators': 80}

boosting = GradientBoostingClassifier(learning_rate = 0.1, n_estimators = 90, subsample = 0.30000000000000004)
boosting.fit(X_train.astype('int64'),y_train)
y_pred_boost_train = boosting.predict(X_train)
boost_mer_train = 1 - metrics.accuracy_score(y_train,y_pred_boost_train)
conf__boost = pd.DataFrame(metrics.confusion_matrix(y_train, y_pred_boost_train))
TP_boost = conf__boost.iloc[0,0]
FP_boost = conf__boost.iloc[1,0]
FN_boost = conf__boost.iloc[0,1]
TN_boost = conf__boost.iloc[1,1]
print("Misclassification Error Rate", boost_mer_train)
print("Training Confusion Matrix \n", metrics.confusion_matrix(y_train, y_pred_boost_train))
print("Recall:", TP_boost/(TP_boost+FN_boost))
print("Specificity:", TN_boost/(TN_boost+FP_boost))
print("Fallout:", FP_boost/(FP_boost + TN_boost))
print("PPV:", TP_boost/(TP_boost + FP_boost))
print("Accuracy:", (TP_boost + TN_boost)/(TP_boost +TN_boost + FP_boost + FN_boost))

y_pred_boost_test = boosting.predict(X_test)
boost_mer_test = 1 - metrics.accuracy_score(y_test,y_pred_boost_test)

print("Misclassification Error Rate", boost_mer_test)
print("Testing Confusion Matrix \n", metrics.confusion_matrix(y_test, y_pred_boost_test))

boost_mer_test = 1 - metrics.accuracy_score(y_test,y_pred_boost_test)
conf__boost1 = pd.DataFrame(metrics.confusion_matrix(y_test, y_pred_boost_test))

TP_boost1 = conf__boost1.iloc[0,0]
FP_boost1 = conf__boost1.iloc[1,0]
FN_boost1 = conf__boost1.iloc[0,1]
TN_boost1 = conf__boost1.iloc[1,1]
print("Misclassification Error Rate", boost_mer_test)
print("Recall:", TP_boost1/(TP_boost1+FN_boost1))
print("Specificity:", TN_boost1/(TN_boost1+FP_boost1))
print("Fallout:", FP_boost1/(FP_boost1 + TN_boost1))
print("PPV:", TP_boost1/(TP_boost1 + FP_boost1))
print("Accuracy:", (TP_boost1 + TN_boost1)/(TP_boost1 +TN_boost1 + FP_boost1 + FN_boost1))

Importance_boosting = pd.DataFrame({'Importance': boosting.feature_importances_*100})
	
Importance_boosting.sort_values(by = 'Importance',
	axis = 0,
	ascending = True). plot(kind = 'barh',
		color = 'r', )

plt.xlabel('Variable Importance')
plt.gca().legend_ = None


c_range = np.arange(1,100,1)


tuned_parameters = [{'C': c_range}]
clf = GridSearchCV(SVC(kernel='linear'), tuned_parameters, cv=10, scoring='accuracy')
clf.fit(X_train, y_train)
print(clf.best_params_)
print(confusion_matrix(y_test, clf.best_estimator_.predict(X_test)))
print(clf.best_estimator_.score(X_test, y_test))

svm = SVC(C=1, kernel='linear', probability=True)
svm.fit(X_train, y_train)
y_pred_linear_train= svm.predict(X_train)
#print(classification_report(y_train, y_pred_linear_train, digits=3))
print(confusion_matrix(y_train, clf.best_estimator_.predict(X_train)))
conf_l = pd.DataFrame(confusion_matrix(y_train, clf.best_estimator_.predict(X_train)))
TPl = conf_l.iloc[0,0]
FPl = conf_l.iloc[1,0]
FNl = conf_l.iloc[0,1]
TNl = conf_l.iloc[1,1]

print("Recall:", TPl/(TPl+FNl))
print("Specificity:", TNl/(TNl+FPl))
print("Fallout:", FPl/(FPl + TNl))
print("PPV:", TPl/(TPl + FPl))
print("Accuracy:", (TPl + TNl)/(TPl +TNl + FPl + FNl))
svm_lr_mer_train = 1 - ((TPl + TNl)/(TPl +TNl + FPl + FNl))
print("Mislcassification Rate:", svm_lr_mer_train)

y_pred_linear_test = svm.predict(X_test)
#print(classification_report(y_test, y_pred_linear_test, digits=3))
print(confusion_matrix(y_test, clf.best_estimator_.predict(X_test)))
conf_l1 = pd.DataFrame(confusion_matrix(y_test, clf.best_estimator_.predict(X_test)))
TPl1 = conf_l1.iloc[0,0]
FPl1 = conf_l1.iloc[1,0]
FNl1 = conf_l1.iloc[0,1]
TNl1 = conf_l1.iloc[1,1]

print("Recall:", TPl1/(TPl1+FNl1))
print("Specificity:", TNl1/(TNl1+FPl1))
print("Fallout:", FPl1/(FPl1 + TNl1))
print("PPV:", TPl1/(TPl1 + FPl1))
print("Accuracy:", (TPl1 + TNl1)/(TPl1 +TNl1 + FPl1 + FNl1))
svm_lr_mer_test = 1 - ((TPl1 + TNl1)/(TPl1 +TNl1 + FPl1 + FNl1))
print("Mislcassification Rate:", svm_lr_mer_test)

c_range_poly = np.arange(1,100,1)
gamma_range_poly = np.arange(0.01,0.05,0.01)
degree_range = np.arange(1,10,1)
tuned_parameters_poly = [{'C': c_range_poly,
                     'degree': degree_range,
                     'gamma': gamma_range_poly}]


clf_poly = GridSearchCV(SVC(kernel='poly'), tuned_parameters_poly, cv=3, scoring='accuracy')
clf_poly.fit(X_train, y_train)
print(clf_poly.best_params_)
print(clf_poly.best_estimator_.score(X_test, y_test))

svm_poly = SVC(C=3, kernel='poly', gamma=0.03, degree = 1, probability=True)
svm_poly.fit(X_train, y_train)


y_pred_poly_train = svm_poly.predict(X_train)
#print(classification_report(y_train, y_pred_poly_train, digits=3))
print(confusion_matrix(y_train, clf_poly.best_estimator_.predict(X_train)))
conf_poly = pd.DataFrame(confusion_matrix(y_train, clf_poly.best_estimator_.predict(X_train)))
TP_poly = conf_poly.iloc[0,0]
FP_poly = conf_poly.iloc[1,0]
FN_poly = conf_poly.iloc[0,1]
TN_poly = conf_poly.iloc[1,1]

print("Recall:", TP_poly/(TP_poly+FN_poly))
print("Specificity:", TN_poly/(TN_poly+FP_poly))
print("Fallout:", FP_poly/(FP_poly + TN_poly))
print("PPV:", TP_poly/(TP_poly + FP_poly))
print("Accuracy:", (TP_poly + TN_poly)/(TP_poly +TN_poly + FP_poly + FN_poly))
svm_poly_mer_train = 1 - ((TP_poly + TN_poly)/(TP_poly +TN_poly + FP_poly + FN_poly))
print("Mislcassification Rate:", svm_poly_mer_train)


y_pred_poly_test = svm_poly.predict(X_test)
#print(classification_report(y_test, y_pred_poly_test, digits=3))
print(confusion_matrix(y_test, clf_poly.best_estimator_.predict(X_test)))
conf__poly1 = pd.DataFrame(confusion_matrix(y_test, clf_poly.best_estimator_.predict(X_test)))
TP_poly1 = conf__poly1.iloc[0,0]
FP_poly1 = conf__poly1.iloc[1,0]
FN_poly1 = conf__poly1.iloc[0,1]
TN_poly1 = conf__poly1.iloc[1,1]

print("Recall:", TP_poly1/(TP_poly1+FN_poly1))
print("Specificity:", TN_poly1/(TN_poly1+FP_poly1))
print("Fallout:", FP_poly1/(FP_poly1 + TN_poly1))
print("PPV:", TP_poly1/(TP_poly1 + FP_poly1))
print("Accuracy:", (TP_poly1 + TN_poly1)/(TP_poly1 +TN_poly1 + FP_poly1 + FN_poly1))
svm_poly_mer_test = 1 - ((TP_poly1 + TN_poly1)/(TP_poly1 +TN_poly1 + FP_poly1 + FN_poly1))
print("Mislcassification Rate:", svm_poly_mer_test)


c_range_rbf = np.arange(1,100,1)
gamma_range_rbf = np.arange(0.01,0.05,0.01)
tuned_parameters_rbf = [{'C': c_range_rbf,
                     'gamma': gamma_range_rbf}]


clf_rbf = GridSearchCV(SVC(kernel='rbf'), tuned_parameters_rbf, cv=3, scoring='accuracy')
clf_rbf.fit(X_train, y_train)
print(clf_rbf.best_params_)
print(clf_rbf.best_estimator_.score(X_test, y_test))


svm_rbf = SVC(C=5, kernel='rbf', gamma=0.01, probability = True)
svm_rbf.fit(X_train, y_train)


y_pred_rbf_train = svm_rbf.predict(X_train)
#print(classification_report(y_train, y_pred_rbf_train, digits=3))
print(confusion_matrix(y_train, clf_rbf.best_estimator_.predict(X_train)))
conf_rbf = pd.DataFrame(confusion_matrix(y_train, clf_rbf.best_estimator_.predict(X_train)))
TP_rbf = conf_rbf.iloc[0,0]
FP_rbf = conf_rbf.iloc[1,0]
FN_rbf = conf_rbf.iloc[0,1]
TN_rbf = conf_rbf.iloc[1,1]

print("Recall:", TP_rbf/(TP_rbf+FN_rbf))
print("Specificity:", TN_rbf/(TN_rbf+FP_rbf))
print("Fallout:", FP_rbf/(FP_rbf + TN_rbf))
print("PPV:", TP_rbf/(TP_rbf + FP_rbf))
print("Accuracy:", (TP_rbf + TN_rbf)/(TP_rbf +TN_rbf + FP_rbf + FN_rbf))
svm_rbf_mer_train = 1 - ((TP_rbf + TN_rbf)/(TP_rbf +TN_rbf + FP_rbf + FN_rbf))
print("Mislcassification Rate:", svm_rbf_mer_train)


y_pred_rbf_test = svm_rbf.predict(X_test)
#print(classification_report(y_test, y_pred_rbf_test, digits=3))
print(confusion_matrix(y_test, clf_rbf.best_estimator_.predict(X_test)))
conf__rbf1 = pd.DataFrame(confusion_matrix(y_test, clf_rbf.best_estimator_.predict(X_test)))
TP_rbf1 = conf__rbf1.iloc[0,0]
FP_rbf1 = conf__rbf1.iloc[1,0]
FN_rbf1 = conf__rbf1.iloc[0,1]
TN_rbf1 = conf__rbf1.iloc[1,1]

print("Recall:", TP_rbf1/(TP_rbf1+FN_rbf1))
print("Specificity:", TN_rbf1/(TN_rbf1+FP_rbf1))
print("Fallout:", FP_rbf1/(FP_rbf1 + TN_rbf1))
print("PPV:", TP_rbf1/(TP_rbf1 + FP_rbf1))
print("Accuracy:", (TP_rbf1 + TN_rbf1)/(TP_rbf1 +TN_rbf1 + FP_rbf1 + FN_rbf1))
svm_rbf_mer_test = 1 - ((TP_rbf1 + TN_rbf1)/(TP_rbf1 +TN_rbf1 + FP_rbf1 + FN_rbf1))
print("Mislcassification Rate:", svm_rbf_mer_test)


#linear kernel
y_pred_linear_prob = svm.predict_proba(X_test)
fpr_lr, tpr_lr, thresholds = roc_curve(y_test, y_pred_linear_prob[: ,  1], pos_label= 1)
print("SVMClassifier (linear kernel): {0}".format(auc(fpr_lr,tpr_lr)))
#polynomial kernel
y_pred_poly_prob = svm_poly.predict_proba(X_test)
fpr_poly, tpr_poly, thresholds_poly = roc_curve(y_test, y_pred_poly_prob[:,1], pos_label= 1)
print("SVMClassifier (polynomial kernel): {0}".format(auc(fpr_poly,tpr_poly)))
#rbf kernel
y_pred_rbf_prob = svm_rbf.predict_proba(X_test)
fpr_rbf, tpr_rbf, thresholds_rbf = roc_curve(y_test, y_pred_rbf_prob[:,1], pos_label= 1)
print("SVMClassifier (rbf kernel): {0}".format(auc(fpr_rbf,tpr_rbf)))
#classification tree
y_pred_clf_prob = linear.predict_proba(X_test)
fpr_clf, tpr_clf, thresholds_clf = roc_curve(y_test, y_pred_clf_prob[:,1], pos_label= 1)
print("Classification Tree: {0}".format(auc(fpr_clf,tpr_clf)))
#bagging
y_pred_bag_prob = bagging.predict_proba(X_test)
fpr_bag, tpr_bag, thresholds_bag = roc_curve(y_test, y_pred_bag_prob[:,1], pos_label= 1)
print("Bagging: {0}".format(auc(fpr_bag,tpr_bag)))
#random forest
y_pred_rf_prob = rforest.predict_proba(X_test)
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, y_pred_rf_prob[:,1], pos_label= 1)
print("Random Forest: {0}".format(auc(fpr_rf,tpr_rf)))
#boosting
y_pred_boost_prob = boosting.predict_proba(X_test)
fpr_boost, tpr_boost, thresholds_boost = roc_curve(y_test, y_pred_boost_prob[:,1], pos_label= 1)
print("Random Forest: {0}".format(auc(fpr_boost,tpr_boost)))
# Plot ROC curve now
fig = plt.figure(figsize=(12,10))
ax = fig.add_subplot(111)
# Connect diagonals
ax.plot([0, 1], [0, 1], ls="--")
# Labels etc
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC curve')
# Set graph limits
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.0])
# Plot each graph now
#ax.plot([fpr_lr, fpr_poly, fpr_rbf],[tpr_lr, tpr_poly, tpr_rbf], label = ["lr","poly","rbf"])
#ax.plot(fpr_lr, tpr_lr, 'linear', fpr_poly, tpr_poly, 'poly', fpr_rbf, tpr_rbf, 'rbf')
ax.plot(fpr_lr,tpr_lr,label = 'linear')
ax.plot(fpr_poly,tpr_poly,label='poly')
ax.plot(fpr_rbf,tpr_rbf,label='rbf')
ax.plot(fpr_clf,tpr_clf,label='clf_tree')
ax.plot(fpr_bag,tpr_bag,label='bagging')
ax.plot(fpr_rf,tpr_rf,label='random_forest')
ax.plot(fpr_boost,tpr_boost,label='boosting')

ax.plot()
# Set legend and show plot
ax.legend(loc="lower right")
plt.show()

#Best Classifier
ct_mer_avg = (ct_mer_train + ct_mer_test)/2
bag_mer_avg = (bag_mer_train + bag_mer_test)/2
rf_mer_avg = (rf_mer_train + rf_mer_test)/2
boost_mer_avg = (boost_mer_train + boost_mer_test)/2
svm_lr_mer_avg = (svm_lr_mer_train + svm_lr_mer_test)/2
svm_poly_mer_avg = (svm_poly_mer_train + svm_poly_mer_test)/2
svm_rbf_mer_avg = (svm_rbf_mer_train + svm_rbf_mer_test)/2
print("Classification Tree MER:", ct_mer_avg)
print("Bagging MER", bag_mer_avg)
print("Random Forest MER", rf_mer_avg)
print("Boosting MER", boost_mer_avg)
print("SVM linear MER", svm_lr_mer_avg)
print("SVM poly MER", svm_poly_mer_avg)
print("SVM_rbf_MER", svm_rbf_mer_avg)


