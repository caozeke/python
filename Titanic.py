#coding:utf-8
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import sklearn.preprocessing as preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import csv

data = pd.read_csv('C:/Users/Administrator/Desktop/train.csv')

age_is_null = pd.isnull(data['Age'])
data['Age'] = data['Age'].fillna(data['Age'].median())
data.loc[data['Sex'] == 'male','Sex'] = 0
data.loc[data['Sex'] == 'female','Sex'] = 1
data.drop(['Name','PassengerId','Ticket','Cabin','Embarked'],axis=1,inplace=True)
data['Age']= preprocessing.scale(data['Age'],axis=0,with_mean=True,with_std=True,copy=True)
data['Fare'] = preprocessing.scale(data['Fare'],with_mean=True,with_std=True,copy=True)
#print data.head()
x_train,x_test,y_train,y_test = train_test_split(data.drop('Survived',axis=1),data['Survived'],test_size=0.2)

alg = LogisticRegression()
alg.fit(x_train,y_train)
y_pred = alg.predict(x_test)
answer = alg.predict_proba(x_test)[:,1]
precision, recall,thresholds = precision_recall_curve(y_test,answer)

plt.plot(recall,precision,lw=1)
#plt.show()



data2 = pd.read_csv('C:/Users/Administrator/Desktop/test.csv')
data2['Age'] = data2['Age'].fillna(data2['Age'].median())
data2['Fare'] = data2['Fare'].fillna(8)


data2.loc[data2['Sex'] == 'male','Sex'] = 0
data2.loc[data2['Sex'] == 'female','Sex'] = 1
data2.drop(['Name','PassengerId','Ticket','Cabin','Embarked'],axis=1,inplace=True)
data2 = data2.dropna()
data2['Age']= preprocessing.scale(data2['Age'],axis=0,with_mean=True,with_std=True,copy=True)
data2['Fare'] = preprocessing.scale(data2['Fare'],with_mean=True,with_std=True,copy=True)

predict = alg.predict(data2)

re = pd.DataFrame(predict,index=range(892,1310))
print re
