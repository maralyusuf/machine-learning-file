import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

dataset = pd.read_csv("dataset.csv")

x = dataset.iloc[:,:1].values
y = dataset.iloc[:,1:].values


for i in range(1,9):
    poly = PolynomialFeatures(degree=i)

    x_poly =poly.fit_transform(x)

    lr = LinearRegression()
    lr.fit(x_poly,y)


    plt.title("degree : "+str(i))
    plt.plot(x,lr.predict(x_poly))
    plt.scatter(x,y,color="red")
    plt.show()
