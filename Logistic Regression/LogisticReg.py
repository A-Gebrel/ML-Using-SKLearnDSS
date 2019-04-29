#  LogisticReg.py made for the DSS project.
#  Written by A.E.A.E, To be committed on Github

#  Imports
from random import randint
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
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
pred = classifier.predict(x_test[0].reshape(1, -1))  # Preparing Pred. incase user picked 1
print("predicted: ", pred[0], ", Actual result: ", y_test[0])

#  Checking Accuracy
acc2 = classifier.score(x_train, y_train)
print("Model Accuracy(train):", acc2)
acc1 = classifier.score(x_test, y_test)
print("Model Accuracy(test):", acc1)

#  Plotting the results
plt.plot(classifier.coef_.T, 'o')
plt.show()
