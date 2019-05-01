#  knn.py made for the DSS project.
#  Written by A.E.A.E, To be committed on Github

#  Imports
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#  Importing our dataset
digits = load_digits()

#  Allocating x,y to our data,target (respectively)
x = digits.data
y = digits.target

#  Creating our training/test sets.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

#  Creating our classifying module.
classifier = KNeighborsClassifier(n_neighbors=4, metric='minkowski', p=2)
classifier.fit(x_train, y_train)

#  Running Predictions
pred = classifier.predict(x_test[0].reshape(1, -1))  # Preparing Pred. incase user picked 1
print("predicted: ", pred[0], ", Actual result: ", y_test[0])

acc2 = classifier.score(x_train, y_train)
print("Model Accuracy(train):", acc2)
acc1 = classifier.score(x_test, y_test)
print("Model Accuracy(test):", acc1)

neighbors_settings = range(1, 11)
training_accuracy = []
test_accuracy = []

for n_neighbors in neighbors_settings:
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)  # Create KNN and specify the K
    clf.fit(x_train, y_train)  # Train KNN
    training_accuracy.append(clf.score(x_train, y_train))  # Score the model based on the trained data
    test_accuracy.append(clf.score(x_test, y_test))  # Score the model based on the test data

# Visualize results - to help with deciding which n_neigbors yields the best results (n_neighbors=6, in this case)
'''
plt.plot(neighbors_settings, training_accuracy, label='Accuracy of the training set')
plt.plot(neighbors_settings, test_accuracy, label='Accuracy of the test set')
plt.title('Error Rate KNN')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
plt.show()'''