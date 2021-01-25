import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import svm, metrics

torso = pd.read_csv('torso_design2_6.csv', encoding="utf-8", header = 0)

target = torso.columns[-1]
X = torso.drop(target, axis = 1)
y = torso.drop(X, axis = 1)

# List storing accuracies for different train-test splits
thisList = list()

# Create a svm Classifier
clf = svm.SVC(kernel='rbf')  # Radial Basis Function Kernel

# Loops through different train-test splits and stores accuracy in list
for x in np.arange(0.1, 0.9, 0.1):
    X_train, X_test, y_train, y_test = train_test_split(X, y.values.ravel(), test_size=x, random_state=109)

    # Train the model using the training sets
    clf.fit(X_train, y_train)

    # Predict the response for test dataset
    y_pred = clf.predict(X_test)

    thisList.append(metrics.accuracy_score(y_test, y_pred))

    # Model Accuracy: how often is the classifier correct?
    # print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# Loop through accuracy list
for x in thisList:
    print(x)