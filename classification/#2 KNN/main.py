from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix


iris = datasets.load_iris()

x = iris["data"]
y = iris["target"]


knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x,y)

y_pred = knn.predict(x)

cn = confusion_matrix(y,y_pred)
print(cn)
