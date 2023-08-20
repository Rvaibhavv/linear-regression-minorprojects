import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.utils import shuffle
import sklearn


data=pd.read_csv("student-mat.csv",sep=";")
data=data[["G1","G2","G3","studytime","failures","absences"]]
print(data.head())

predict="G3"
x=np.array(data.drop([predict],1))
y=np.array(data[predict])
print(x)

x_train, x_test, y_train,  y_test=sklearn.model_selection.train_test_split(x,y,test_size=0.1)
linear=linear_model.LinearRegression()
linear.fit(x_train, y_train)
acc=linear.score(x_test, y_test)
print(acc)

print(linear.coef_)
print(linear.intercept_)

predictions=linear.predict(x_test)
for i in range(len(predictions)):
  print(predictions[i],x_test[i],y_test[i])
