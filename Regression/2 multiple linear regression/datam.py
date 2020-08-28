import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score
import matplotlib.pyplot as plt

dataset = pd.read_csv("dataset.csv")

x = dataset.iloc[:,:2].values
y = dataset.iloc[:,2:].values

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.22,random_state=0)

liner = LinearRegression()

liner.fit(x_train,y_train)

y_pred = liner.predict(x_test)



for i in range(0,len(y_pred)):
    if y_pred[i] >= 0.5:
        y_pred[i] = 1
    elif y_pred[i] < 0.5:
        y_pred[i] = 0

conf = accuracy_score(y_test,y_pred)
print(conf)

plt.scatter(x[:,:1],y[:])
plt.scatter(x[:,1:],y)

plt.show()
