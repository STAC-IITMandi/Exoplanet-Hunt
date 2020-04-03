#Import scikit-learn, os and sys
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import os
import sys
from sklearn.metrics import confusion_matrix
import os
import sys

#for importing preprocessors
cwd = os.getcwd()
sys.path.insert(1, os.path.dirname(cwd))
from preprocessors import reader

#load data
labels, fluxes = reader.ReadData(os.path.dirname(cwd)+'/data.csv')

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(fluxes, labels, test_size=0.3,random_state=0) # 70% training and 30% test

#Create a svm Classifier
classifier = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
classifier.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = classifier.predict(X_test)

print('Accuracy of SVM classifier on test set: {:.2f}'.format(classifier.score(X_test, y_test)))

confusion_matrix = confusion_matrix(y_test, y_pred)
print('Confusion matrix is')
print(confusion_matrix)

print(classification_report(y_test, y_pred))
