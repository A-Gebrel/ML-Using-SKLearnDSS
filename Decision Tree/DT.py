#  DT.py made for the DSS project.
#  Written by A.E.A.E, To be committed on Github

#  Imports
from sklearn.datasets import load_digits
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

#  Importing our dataset
digits = load_digits()

#  Allocating x,y to our data,target (respectively)
x = digits.data
y = digits.target

#  Creating our training/test sets.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=0)

#  Creating our classifying module.
classifier = DecisionTreeClassifier(max_depth=14, random_state=0)
classifier.fit(x_train, y_train)

#  Running Predictions
pred = classifier.predict(x_test[0].reshape(1, -1))
print("(Testing the module: predicted: ", pred[0], ", Actual result: ", y_test[0])

#  Checking Accuracy
acc1 = classifier.score(x_train, y_train)
print("Model Accuracy(train):", acc1*100)
acc2 = classifier.score(x_test, y_test)
print("Model Accuracy(test):", acc2*100)

