import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix

dataset = pd.read_csv("dataset.csv")

x = dataset.iloc[:,:2].values
y = dataset.iloc[:,2].values

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33)


lr = LogisticRegression()
lr.fit(x_train,y_train)


y_pred = lr.predict(x_test)

cn = confusion_matrix(y_test,y_pred)
print(cn)
