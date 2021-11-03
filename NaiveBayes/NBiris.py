import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import sklearn as sk
from sklearn import datasets
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB


model = GaussianNB()
data = datasets.load_iris()
X = data.data
Y = data.target
model.fit(X, Y)
expected = Y
predicted = model.predict(X)

print(metrics.classification_report(expected, predicted))
conf_matric = metrics.confusion_matrix(expected, predicted)
print(conf_matric)
plt.imshow(conf_matric)
plt.show()



























# if __name__ == '__main__':
#     pass
