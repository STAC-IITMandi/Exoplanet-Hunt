import numpy as np
import pandas as pd

dataset_train = pd.read_csv('exoTrain.csv')
dataset_test = pd.read_csv('exoTest.csv')

dataset = [dataset_train, dataset_test]

y_train = pd.DataFrame(dataset_train['LABEL'])
y_test  = pd.DataFrame(dataset_test['LABEL'])

for i in range(len(dataset)):
    dataset[i] = dataset[i].drop('LABEL', axis = 1)

X_train = dataset[0]
X_test = dataset[1]

#Now we will try all the classification models and then refit them to train data to see which one is the best
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier

clf = LogisticRegression()
clf.fit(X_train, y_train)
y_pred_log_reg = clf.predict(X_test)
acc_log_reg = round( clf.score(X_train, y_train) * 100, 2)
print (str(acc_log_reg) + ' percent') #80.36%

clf = SVC()
clf.fit(X_train, y_train)
y_pred_svc = clf.predict(X_test)
acc_svc = round(clf.score(X_train, y_train) * 100, 2)
acc_svc_pred = round(clf.score(X_test, y_test) * 100, 2)
print (acc_svc) #83.28

clf = LinearSVC()
clf.fit(X_train, y_train)
y_pred_linear_svc = clf.predict(X_test)
acc_linear_svc = round(clf.score(X_train, y_train) * 100, 2)
print (acc_linear_svc) #79.12

clf = KNeighborsClassifier(n_neighbors = 3)
clf.fit(X_train, y_train)
y_pred_knn = clf.predict(X_test)
acc_knn = round(clf.score(X_train, y_train) * 100, 2)
print (acc_knn) #77.44

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred_decision_tree = clf.predict(X_test)
acc_decision_tree = round(clf.score(X_train, y_train) * 100, 2)
print (acc_decision_tree) #87.09

clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
y_pred_random_forest = clf.predict(X_test)
acc_random_forest = round(clf.score(X_train, y_train) * 100, 2)
print (acc_random_forest) #87.09

clf = GaussianNB()
clf.fit(X_train, y_train)
y_pred_gnb = clf.predict(X_test)
acc_gnb = round(clf.score(X_train, y_train) * 100, 2)
print (acc_gnb) #77.78

clf = Perceptron(max_iter=5, tol=None)
clf.fit(X_train, y_train)
y_pred_perceptron = clf.predict(X_test)
acc_perceptron = round(clf.score(X_train, y_train) * 100, 2)
print (acc_perceptron) #78.11

clf = SGDClassifier(max_iter=5, tol=None)
clf.fit(X_train, y_train)
y_pred_sgd = clf.predict(X_test)
acc_sgd = round(clf.score(X_train, y_train) * 100, 2)
print (acc_sgd) #71.72

#Confusion Matrix for random forest classifier
from sklearn.metrics import confusion_matrix
import itertools

clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
y_pred_random_forest_training_set = clf.predict(X_train)
acc_random_forest = round(clf.score(X_train, y_train) * 100, 2)
print ("Accuracy: %i %% \n"%acc_random_forest)
'''
class_names = ['Survived', 'Not Survived']

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_train, y_pred_random_forest_training_set)
np.set_printoptions(precision=2)

print ('Confusion Matrix in Numbers')
print (cnf_matrix)
print ('')

cnf_matrix_percent = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]

print ('Confusion Matrix in Percentage')
print (cnf_matrix_percent)
print ('')

true_class_names = ['True Survived', 'True Not Survived']
predicted_class_names = ['Predicted Survived', 'Predicted Not Survived']

df_cnf_matrix = pd.DataFrame(cnf_matrix, 
                             index = true_class_names,
                             columns = predicted_class_names)

df_cnf_matrix_percent = pd.DataFrame(cnf_matrix_percent, 
                                     index = true_class_names,
                                     columns = predicted_class_names)

plt.figure(figsize = (15,5))

plt.subplot(121)
sns.heatmap(df_cnf_matrix, annot=True, fmt='d')

plt.subplot(122)
sns.heatmap(df_cnf_matrix_percent, annot=True)
'''
#Comparing Models

models = pd.DataFrame({
    'Model': ['Logistic Regression', 'Support Vector Machines', 'Linear SVC', 
              'KNN', 'Decision Tree', 'Random Forest', 'Naive Bayes', 
              'Perceptron', 'Stochastic Gradient Decent'],
    
    'Score': [acc_log_reg, acc_svc, acc_linear_svc, 
              acc_knn,  acc_decision_tree, acc_random_forest, acc_gnb, 
              acc_perceptron, acc_sgd]
    })

models.sort_values(by='Score', ascending=False)
print(models)

#XG boost
from numpy import loadtxt
from xgboost import XGBClassifier


clf = XGBClassifier()
clf.fit(X_train, y_train)
y_pred_xgboost = clf.predict(X_test)
acc_xgboost = round(clf.score(X_train, y_train) * 100, 2)
print (acc_gnb) #77.78

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, y_pred_decision_tree))
print(accuracy_score(y_test, y_pred_gnb))
print(accuracy_score(y_test, y_pred_knn))
print(accuracy_score(y_test, y_pred_linear_svc))
print(accuracy_score(y_test, y_pred_log_reg))
print(accuracy_score(y_test, y_pred_perceptron))
print(accuracy_score(y_test, y_pred_random_forest))
print(accuracy_score(y_test, y_pred_sgd))
print(accuracy_score(y_test, y_pred_xgboost))
print(accuracy_score(y_test, y_pred_svc))

models = pd.DataFrame({
    'Model': ['Logistic Regression', 'Support Vector Machines', 'Linear SVC', 
              'KNN', 'Decision Tree', 'Random Forest', 'Naive Bayes', 
              'Perceptron', 'Stochastic Gradient Decent', 'XGBoos'],
    
    'Score': [accuracy_score(y_test, y_pred_log_reg), accuracy_score(y_test, y_pred_svc), accuracy_score(y_test, y_pred_linear_svc), 
              accuracy_score(y_test, y_pred_knn),  accuracy_score(y_test, y_pred_decision_tree), accuracy_score(y_test, y_pred_random_forest), accuracy_score(y_test, y_pred_gnb), 
              accuracy_score(y_test, y_pred_perceptron), accuracy_score(y_test, y_pred_sgd), accuracy_score(y_test, y_pred_xgboost)]
    })
import sklearn.metrics as m
cf_svc = m.confusion_matrix(y_test, y_pred_svc)
cf_knn = m.confusion_matrix(y_test, y_pred_knn)

re_svc = m.recall_score(y_test, y_pred_svc)
re_knn = m.recall_score(y_test, y_pred_knn)

pr_svc = m.precision_score(y_test, y_pred_svc)
pr_knn = m.precision_score(y_test, y_pred_knn)

m.classification_report(y_test, y_pred_svc)
