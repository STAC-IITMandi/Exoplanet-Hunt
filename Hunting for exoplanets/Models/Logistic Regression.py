import numpy as np
import pandas as pd
import os

os.chdir('../')
os.chdir('Data/')
dataset_train = pd.read_csv('exoTrain.csv')
dataset_test = pd.read_csv('exoTest.csv')

dataset = [dataset_train, dataset_test]

y_train = dataset_train['LABEL']
y_test  = dataset_test['LABEL']

for i in range(len(dataset)):
    dataset[i] = dataset[i].drop('LABEL', axis = 1)

from sklearn.preprocessing import StandardScaler

scale = StandardScaler()
for i in range(len(dataset)):
    scale.fit(dataset[i])
    dataset[i] = scale.transform(dataset[i])

X_train = dataset[0]
X_test = dataset[1]

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import sklearn.metrics as m


def model_log_reg():
    clf = LogisticRegression(solver = 'lbfgs', max_iter = 2000)
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    print('\n\n')
    print('F1 score : ',accuracy_score(y_test, y_pred))
    print('Confusion Matrix : \n',m.confusion_matrix(y_test, y_pred))
    print('Precision : ',m.precision_score(y_test, y_pred))
    print('Recall : ',m.recall_score(y_test, y_pred))
    return y_pred

y_pred = model_log_reg()