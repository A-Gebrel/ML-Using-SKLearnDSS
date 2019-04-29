#  LogisticReg.py made for the DSS project.
#  Written by A.E.A.E, To be committed on Github

#  Imports
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score

#  Actual code
digits = load_digits()

#  Allocating x,y to our data,target (respectively)
x = digits.data
y = digits.target

#  Creating our training/test sets.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

#  Classifying the LogisticRegression % fitting to train
classifier = LogisticRegression(random_state=0)
classifier.fit(x_train, y_train)

#  Pick your Option
print("1-\t Print Accuracy only")
print("2-\t Print Predictions & Accuracy")
k1 = input("Select your Option: ")

#  Running Predictions
if k1 == "2":
    i = 0
    while i < 360:
        pred = classifier.predict(x_test[i].reshape(1, -1))
        if pred[0] == y_test[i]: x = "Y"
        else: x = "N"
        print("[",i,"]predicted: ", pred[0], ", Actual result: ", y_test[i], x)
        i += 1

#  Checking Accuracy
acc = classifier.score(x_test, y_test)
print("Model Accuracy:", acc)
