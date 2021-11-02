import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import sklearn.linear_model as skl_lm

bd = datasets.load_boston()
names = bd['feature_names']
bd.columns = names
X = pd.DataFrame(bd.data)
X.columns = names
y = pd.DataFrame(bd.target)
y.columns = np.array(["MEDV"])
X.head()
y.head()
RM = X['RM']

#Prudcuce plot of 'MEDV' vs 'RM'
ax = plt.gca()
ax.scatter(RM, y)
plt.xlabel('RM')
plt.ylabel('MDEV')

df = pd.DataFrame(RM)
df['MDEV'] = y
0.8 * len(df)
train_df = df.sample(405, random_state=1)
test_df = df[~df.isin(train_df)].dropna(how = 'all')

X_train = train_df['RM'].values.reshape(-1,1)
y_train = train_df['MDEV']
X_test = test_df['RM'].values.reshape(-1,1)
y_test = test_df['MDEV']

lm = skl_lm.LinearRegression()
model = lm.fit(X_train, y_train)
pred = model.predict(X_test)

from sklearn.metrics import mean_squared_error

MSE = mean_squared_error(y_test, pred)

print("Linear Regression MSE: ", MSE)

#Kfold cv

from sklearn.model_selection import KFold, cross_val_score

cv_k = KFold(n_splits = 10, random_state = 1, shuffle=True)

scores_k = cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv = cv_k, n_jobs=1)

print("Using KFold:  Folds: " + str(len(scores_k)) +", MSE: " + str(np.mean(np.abs(scores_k))) + ", STD: " + str(np.std(scores_k)))

print("Cross-validated scores:", np.abs(scores_k))

from sklearn.model_selection import cross_val_predict
predictions = cross_val_predict(model, X, y, cv=cv_k, n_jobs=1)
#plt.scatter(y,predictions)

#LOOCV

model = lm.fit(X_train, y_train)
from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()
X = df['RM'].values.reshape(-1,1)
y = df['MDEV'].values.reshape(-1,1)
loo.get_n_splits(X)

cv_loo = KFold(n_splits = len(X), random_state=None, shuffle=True)

scores_loo = cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=cv_loo, n_jobs=1)

print("Using LOOCV  Folds: " + str(len(scores_loo)), ", MSE: " + str(np.mean(np.abs(scores_loo))) + ", STD: " +str(np.std(scores_loo)))


def bt(df, B):
    sample_R=[]
    data = []
    for i in range(B):
        for j in range(10):
            x = np.random.choice(503, 1)
            n = df.iloc[x,:]
            data.append(dict(n))
        y = pd.DataFrame(data)
        z = y.astype(float)
        corr = z.corr(method='pearson')
        corr1 = corr.iloc[0,1]
        sample_R.append(corr1)
        data=[]
    return(sample_R)
        


test = bt(df,50)
graph = pd.DataFrame(test)
graph.plot.hist(grid = True,bins = 10, rwidth=1)
plt.title('Distrubution of R')
plt.xlabel('Sample')
plt.ylabel('R')
plt.grid(axis='y', alpha=0.75)

print("Median estimate: ", np.median(test))
print("Average estimate: ", np.mean(test))
print("Standard Error estimate: ", np.std(test))

basic = np.quantile(test,(0.025,0.975))
print("95% confidence interval: ", basic)

def bt_beta(df,B):
  data = []
  sample_beta = []
  sample_intercept = []
  lm = skl_lm.LinearRegression()
  for i in range(B):
    for j in range(10):
      x = np.random.choice(503, 1)
      n = df.iloc[x,:]
      data.append(dict(n))
    names = ['RM', 'MDEV']
    frame = pd.DataFrame(data = data, columns = names)
    flo = frame.astype(float)
    X = np.array(flo['RM']).reshape(-1,1)
    y = np.array(flo['MDEV']).reshape(-1,1)
    n = lm.fit(X,y)
    c = n.coef_
    inter = n.intercept_
    sample_intercept.append(inter)
    sample_beta.append(c)
    data = []
  #result = {'Intercept': [sample_intercept], 'Beta': [sample_beta]}
  int_result = pd.DataFrame(sample_intercept)
  results= pd.DataFrame(int_result)
  beta_result = np.array(sample_beta).reshape(-1,1)
  beta_formatted = pd.DataFrame(beta_result)
  results['Beta'] = pd.DataFrame(beta_formatted)
  return(results)
 


X = pd.DataFrame(df['RM'])
y = pd.DataFrame(df['MDEV'])
test2 = bt_beta(df, 50)
intercepts = test2.iloc[:,0]
betas = test2['Beta']
intercept_result = np.mean(intercepts)
beta_result = np.mean(betas)
print("Intercept Estimate: ", intercept_result )
print("Beta Estimate", beta_result)

def bt_beta_SE(df, B):
    data = np.zeros((B,2))
    sample_betaSE = []
    sample_intSE = []
    for i in range(B):
        for j in range(10):
            x = np.random.choice(50, 1)
            n = df.iloc[x,:]
            data[i] = n
        names = ['Intercept', 'Beta']
        frame = pd.DataFrame(data = data, columns = names)
        flo = frame.astype(float)
        X = np.array(flo['Intercept']).reshape(-1,1)
        y = np.array(flo['Beta']).reshape(-1,1)
        int_se = np.std(X)
        beta_se = np.std(y)
        sample_betaSE.append(beta_se)
        sample_intSE.append(int_se)
    int_result = np.array(sample_intSE)
    int_formatted = pd.DataFrame(int_result)
    results= pd.DataFrame(int_formatted)
    beta_result = np.array(sample_betaSE)
    beta_formatted = pd.DataFrame(beta_result)
    results['Beta'] = pd.DataFrame(beta_formatted)
    return(results)

#k = test2.iloc[2, :]
#k_std = np.std(k)

test3 = bt_beta_SE(test2, 50)
intercept_se = test3.iloc[:,0]
beta_se = test3['Beta']
intercept_se_avg = np.mean(intercept_se)
beta_se_avg = np.mean(beta_se)
print("Intercept SE Estimate: ", intercept_se_avg)
print("Beta SE Estimate", beta_se_avg)

import statsmodels.api as sm
lr = pd.DataFrame()
lr['x'] = df['RM']
lr['y'] = df['MDEV']

lm = sm.OLS.from_formula('y~x', lr)
result = lm.fit()
print(result.summary())

#confidence interval

lower_bound = beta_result - 2*beta_se_avg
upper_bound = beta_result + 2*beta_se_avg


CI = np.array([lower_bound,upper_bound])
print("Confidence Interval:", CI)