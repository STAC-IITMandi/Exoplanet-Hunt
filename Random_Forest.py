import pandas as pd
import numpy as np
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv("train") #train is location of training data
data = data.dropna(axis='columns', thresh = int(0.5 * len(data)))
data.fillna(data.mean())
col = data.columns
X = data[col[1:]]
y = data['LABEL']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Create the model with 100 trees
model = model = RandomForestClassifier(n_estimators=100, 
                               bootstrap = True,
                               max_features = 'sqrt')
# Fit on training data
model.fit(X_train, y_train)
print(f'Model Accuracy: {model.score(X_test, y_test)}')
