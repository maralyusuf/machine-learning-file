from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import confusion_matrix


iris = load_iris()

x = iris.data
y = iris.target

nb = MultinomialNB()
nb.fit(x,y)


pred = nb.predict(x)


cm = confusion_matrix(y,pred)
print(cm)
