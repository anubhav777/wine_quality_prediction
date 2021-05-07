import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from function import outlier_filter,one_off_filter
from collections import Counter

df=pd.read_csv('winequality-red.csv')

df.columns = df.columns.str.replace(' ', '_')

plt.figure(figsize=(15,10))
cor=df.corr()
sns.heatmap(cor,cmap='coolwarm',annot=True, fmt='.0%')

outl=df.drop('quality',axis=1)

for i in outl.columns:
    plt.hist(df[i])
    plt.title(i)
    plt.show()

df['sulphates'].describe()



IQR=df.sulphates.quantile(0.75) - df.sulphates.quantile(0.25)

print(df[df['sulphates'] > 1.26].index)

# # df.loc[df['sulphates'] > 1.26,'sulphates'] = 1.26
# bla=df.loc[df['sulphates'] > 1.26]

# df.drop(df[df['sulphates'] > 1.26].index,inplace=True)
arr=['sulphates','volatile_acidity','chlorides','citric_acid','total_sulfur_dioxide']

for i in arr:
    outlier_filter(i)

sns.countplot(df['quality'])

df.isnull().sum()

X=df.drop(['fixed_acidity','free_sulfur_dioxide','residual_sugar','pH','quality','total_sulfur_dioxide'],axis=1).values
X.shape

y=df['quality'].values
y

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.3)

from sklearn.linear_model import LinearRegression

lr=LinearRegression()

lr.fit(X_train,y_train)

y_pred=lr.predict(X_test)

from sklearn.metrics import confusion_matrix,precision_score,f1_score,recall_score

cm_decision_tree_regression = confusion_matrix(y_test,np.round(y_pred))
cm_decision_tree_regression

cm_decision_tree_regression = confusion_matrix(y_test,np.round(y_pred))
cm_dt = pd.DataFrame(cm_decision_tree_regression,
                     index = ['3','4','5','6','7','8'], 
                     columns = ['3','4','5','6','7','8'])
sns.heatmap(cm_dt,annot=True,fmt="d")

print(lr.score(X_train,y_train))
print(lr.score(X_test,y_test))

sns.distplot((y_test-y_pred))

from sklearn.metrics import mean_squared_error,r2_score

mean_squared_error(y_test,y_pred, squared=False)

r2_score(y_test,y_pred)

from sklearn.linear_model import Lasso

ls=Lasso(alpha=0.001)

ls.fit(X_train,y_train)

ls_pred=ls.predict(X_test)

print(ls.score(X_train,y_train))
print(ls.score(X_test,y_test))

r2_score(y_test,ls_pred)

mean_squared_error(y_test,ls_pred, squared=False)



one_off_filter(y_pred,y_test)
one_off_filter(dc_pred,y_test)

print(precision_score(y_test,np.round(y_pred), average = 'weighted'),
recall_score(y_test,np.round(y_pred), average = 'weighted'),
f1_score(y_test,np.round(y_pred), average = 'weighted'))


print(precision_score(y_test,np.round(dc_pred), average = 'weighted'),
recall_score(y_test,np.round(dc_pred), average = 'weighted'),
f1_score(y_test,np.round(dc_pred), average = 'weighted'))

y=np.asarray([1 if i>=7 else 0 for i in y])
Counter(y)

X_train,X_test,y_train,y_test=train_test_split(X,y)
Counter(y_train)

Counter(y_test)

from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

#Linear Regression
lr=LogisticRegression(solver='lbfgs')
lr.fit(X_train,y_train)
lr_pred=lr.predict_proba(X_test)
cross_val_score(lr,X_train,y_train,cv=10,scoring='f1')

roc_auc_score(y_test,lr_pred[:,1])

zer=[]
for i in y:
    if i == 0:
        zer.append(i)
unbal=(len(zer)/len(y))*100
unbal


ad=AdaBoostClassifier()
ad.fit(X_train,y_train)
ad_pred=ad.predict_proba(X_test)

roc_auc_score(y_test,ad_pred[:,1])


dt=DecisionTreeClassifier() 
dt.fit(X_train,y_train) 
dt_pred=dt.predict_proba(X_test)

roc_auc_score(y_test,dt_pred[:,1])

cross_val_score(ad,X_train,y_train,cv=10,scoring='f1')