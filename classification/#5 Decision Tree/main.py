from sklearn.datasets import load_digits
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

digits = load_digits()

x = digits.data
y = digits.target

dtc = DecisionTreeClassifier(max_depth=50)
dtc.fit(x,y)

pred = dtc.predict(x)

conf = confusion_matrix(y,pred)

print(conf)
