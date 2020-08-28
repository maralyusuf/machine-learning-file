from sklearn import datasets
from sklearn.svm import SVC
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

digits = datasets.load_digits()

x = digits["data"]
y = digits["target"]

svm = SVC(gamma=0.22)
svm.fit(x,y)

y_pred = svm.predict(x)

print(x.shape)

con = confusion_matrix(y,y_pred)
print(con)
