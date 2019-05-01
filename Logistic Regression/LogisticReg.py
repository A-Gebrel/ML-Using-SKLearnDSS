#  LogisticReg.py made for the DSS project.
#  Written by A.E.A.E, To be committed on Github

#  Imports
from random import randint
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt


#  Importing our dataset
digits = load_digits()

#  Allocating x,y to our data,target (respectively)
x = digits.data
y = digits.target

#  Creating our training/test sets.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=40)

#  Classifying the LogisticRegression % fitting to train
classifier = LogisticRegression(random_state=0)
classifier.fit(x_train, y_train)

#  Pick your Option
print("1-\t Print Accuracy only")
print("2-\t Print Predictions & Accuracy")
# k1 = input("Select your Option: ")
k1 = "1"

#  Running Predictions
pred = classifier.predict(x_test)  # Preparing Pred. incase user picked 1
print("predicted: ", pred[0], ", Actual result: ", y_test[0])


#  Running Probabilities
proba1 = classifier.predict_proba(x_test)

#  Checking Accuracy
acc2 = classifier.score(x_train, y_train)
print("Model Accuracy(train):", acc2)
acc1 = classifier.score(x_test, y_test)
print("Model Accuracy(test):", acc1)

scores = cross_val_score(LogisticRegression(), x, y, scoring='accuracy', cv=10)
print (scores)
print (scores.mean())

#  Plotting the results
'''
plt.scatter(pred, y_test)
plt.title("Logistic Regression Model")
plt.show()
'''