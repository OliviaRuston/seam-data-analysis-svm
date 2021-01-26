import pandas as pd
from sklearn import svm
from sklearn.model_selection import cross_val_score

# Read in csv into a dataframe
seams = pd.read_csv('sample-torso-6.csv', encoding="utf-8", header = 0)

# Define features and targets
target = seams.columns[-1]
X = seams.drop(target, axis = 1)
y = seams.drop(X, axis = 1)

# Classification algorithm set up
clf = svm.SVC(kernel='rbf', C=1, random_state=42)

# Uses 5-fold cross validation, scores prediction accuracy
scores = cross_val_score(clf, X, y.to_numpy().ravel(), cv=5)

print(scores)
