#  all.py made for the DSS project.
#  Written by A.E.A.E, To be committed on Github

#  Imports
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#  Importing our dataset
digits = load_digits()

#  Allocating x,y to our data,target (respectively)
x = digits.data
y = digits.target

#  Creating our training/test sets.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=0)
#  Classifying the LogisticRegression % fitting to train
classifier1 = LogisticRegression(C=0.01, multi_class="auto", random_state=0)
classifier2 = KNeighborsClassifier(n_neighbors=2, metric='minkowski', p=2)
classifier3 = DecisionTreeClassifier(max_depth=14, random_state=0)
classifier1.fit(x_train, y_train)
classifier2.fit(x_train, y_train)
classifier3.fit(x_train, y_train)

#  Running Predictions
print("Running Predictions with classifiers:\n")
pred = classifier1.predict(x_test[0].reshape(1, -1))
print("(Testing LogisticRegression: predicted: ", pred[0], ", Actual result: ", y_test[0])
pred = classifier2.predict(x_test[0].reshape(1, -1))
print("Testing K-NN: predicted: ", pred[0], ", Actual result: ", y_test[0])
pred = classifier3.predict(x_test[0].reshape(1, -1))
print("Testing DecisionTree: predicted: ", pred[0], ", Actual result: ", y_test[0])

print("========================")

#  Checking Accuracy
print("Checking Accuracy with classifiers\n")
acc1 = classifier1.score(x_train, y_train)
print("[LogisticReg] Model Accuracy(train):", acc1*100)
acc2 = classifier1.score(x_test, y_test)
print("[LogisticReg] Model Accuracy(test):", acc2*100)
print("========================")
acc1 = classifier2.score(x_train, y_train)
print("[K-NN] Model Accuracy(train):", acc1*100)
acc2 = classifier2.score(x_test, y_test)
print("[K-NN] Model Accuracy(test):", acc2*100)
print("========================")
acc1 = classifier3.score(x_train, y_train)
print("[DecisionTree] Model Accuracy(train):", acc1*100)
acc2 = classifier3.score(x_test, y_test)
print("[DecisionTree] Model Accuracy(test):", acc2*100)

test_accuracy = []
ctest = np.arange(0.1, 5, 0.1)

for c in ctest:
    clf = LogisticRegression(solver='liblinear', C=c, multi_class="auto", random_state=0)
    clf.fit(x,y)
    test_accuracy.append(clf.score(x_test, y_test))


plt.plot(ctest, test_accuracy, label='Accuracy of the test set')
plt.ylabel('Accuracy')
plt.legend()
plt.show()