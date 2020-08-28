from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix


digits = load_digits()

x = digits.data
y = digits.target


rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(x,y)

pred = rfc.predict(x)


acc = accuracy_score(y,pred)
print("accuracy : "+str(acc))
