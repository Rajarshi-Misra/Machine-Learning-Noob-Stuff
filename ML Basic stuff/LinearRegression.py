import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
diabetes = datasets.load_boston() #Boston dataset
diabetes_X  =  diabetes.data[:,np.newaxis,2]
diabetes_X_train  = diabetes_X[:-30]
diabetes_X_test = diabetes_X[-30:]
diabetes_y_train  = diabetes.target[:-30]
diabetes_y_test = diabetes.target[-30:]
model = linear_model.LinearRegression()
model.fit(diabetes_X_train, diabetes_y_train)
diabetes_y_predicted = model.predict(diabetes_X_test)
plt.scatter(diabetes_X_test, diabetes_y_test)
plt.plot(diabetes_X_test, diabetes_y_predicted)
plt.show()
print(diabetes.keys())