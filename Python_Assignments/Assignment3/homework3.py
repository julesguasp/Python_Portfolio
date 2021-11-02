# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 19:14:08 2021

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

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data"
data = pd.read_csv(url)
data.head()
data
df = pd.DataFrame(data)
X = df.iloc[:,0:60]
X.head()
y = df.iloc[:,60]
y.head()
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25, random_state=0)

c_range = np.arange(50,60,0.5)
gamma_range = np.arange(0.01,0.05,0.01)
tuned_parameters = [{'C': c_range,
                     'gamma': gamma_range}]
clf = GridSearchCV(SVC(kernel='linear'), tuned_parameters, cv=10, scoring='accuracy')
clf.fit(X_train, y_train)
clf.best_params_
print(confusion_matrix(y_test, clf.best_estimator_.predict(X_test)))
print(clf.best_estimator_.score(X_test, y_test))

svm = SVC(C=54, kernel='linear', gamma=0.01, probability=True)
svm.fit(X_train, y_train)
y_pred_linear_train= svm.predict(X_train)
print(classification_report(y_train, y_pred_linear_train, digits=3))
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
print("Recall:", TPl/(TPl+FNl))
print("Specificity:", TNl/(TNl+FPl))
print("Fallout:", FPl/(FPl + TNl))
print("PPV:", TPl/(TPl + FPl))
print("Accuracy:", (TPl + TNl)/(TPl +TNl + FPl + FNl))
print("Misclassification Rate:", 1 - ((TPl + TNl)/(TPl +TNl + FPl + FNl)))

y_pred_linear_test = svm.predict(X_test)
print(classification_report(y_test, y_pred_linear_test, digits=3))
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
print("Misclassification Rate:", 1 - ((TPl1 + TNl1)/(TPl1 +TNl1 + FPl1 + FNl1)))

c_range_poly = np.arange(1,100,1)
gamma_range_poly = np.arange(0.01,0.05,0.01)
tuned_parameters_poly = [{'C': c_range_poly,
                     'gamma': gamma_range_poly}]


clf_poly = GridSearchCV(SVC(kernel='poly'), tuned_parameters_poly, cv=10, scoring='accuracy')
clf_poly.fit(X_train, y_train)
print(clf_poly.best_params_)
print(clf_poly.best_estimator_.score(X_test, y_test))

svm_poly = SVC(C=93, kernel='poly', gamma=0.04, probability=True)
svm_poly.fit(X_train, y_train)


y_pred_poly_train = svm_poly.predict(X_train)
print(classification_report(y_train, y_pred_poly_train, digits=3))
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
print("Misclassification Rate:", 1 - ((TP_poly + TN_poly)/(TP_poly +TN_poly + FP_poly + FN_poly)))


y_pred_poly_test = svm_poly.predict(X_test)
print(classification_report(y_test, y_pred_poly_test, digits=3))
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
print("Misclassification Rate:", 1 - ((TP_poly1 + TN_poly1)/(TP_poly1 +TN_poly1 + FP_poly1 + FN_poly1)))


c_range_rbf = np.arange(1,100,1)
gamma_range_rbf = np.arange(0.01,0.05,0.01)
tuned_parameters_rbf = [{'C': c_range_rbf,
                     'gamma': gamma_range_rbf}]


clf_rbf = GridSearchCV(SVC(kernel='rbf'), tuned_parameters_rbf, cv=10, scoring='accuracy')
clf_rbf.fit(X_train, y_train)
print(clf_rbf.best_params_)
print(clf_rbf.best_estimator_.score(X_test, y_test))

#Best Parameters C:27, Gamma: 0.04
svm_rbf = SVC(C=27, kernel='rbf', gamma=0.04, probability = True)
svm_rbf.fit(X_train, y_train)


y_pred_rbf_train = svm_rbf.predict(X_train)
print(classification_report(y_train, y_pred_rbf_train, digits=3))
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
print("Misclassification Rate:", 1 - ((TP_rbf + TN_rbf)/(TP_rbf +TN_rbf + FP_rbf + FN_rbf)))


y_pred_rbf_test = svm_rbf.predict(X_test)
print(classification_report(y_test, y_pred_rbf_test, digits=3))
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
print("Misclassification Rate:", 1 - ((TP_rbf1 + TN_rbf1)/(TP_rbf1 +TN_rbf1 + FP_rbf1 + FN_rbf1)))

y_test_binary = y_test.replace('R', 1)
y_test_binary = y_test_binary.replace('M', 0)
#linear kernel
y_pred_linear_prob = svm.predict_proba(X_test)
fpr_lr, tpr_lr, thresholds = roc_curve(y_test_binary, y_pred_linear_prob[: ,  1], pos_label= 1)
print("SVMClassifier (linear kernel): {0}".format(auc(fpr_lr,tpr_lr)))
#polynomial kernel
y_pred_poly_prob = svm_poly.predict_proba(X_test)
fpr_poly, tpr_poly, thresholds_poly = roc_curve(y_test_binary, y_pred_poly_prob[:,1], pos_label= 1)
print("SVMClassifier (polynomial kernel): {0}".format(auc(fpr_poly,tpr_poly)))
#rbf kernel
y_pred_rbf_prob = svm_rbf.predict_proba(X_test)
fpr_rbf, tpr_rbf, thresholds_rbf = roc_curve(y_test_binary, y_pred_rbf_prob[:,1], pos_label= 1)
print("SVMClassifier (rbf kernel): {0}".format(auc(fpr_rbf,tpr_rbf)))

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

# Set legend and show plot
ax.legend(loc="lower right")
plt.show()