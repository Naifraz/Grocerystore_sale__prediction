import numpy as np
import pandas as pd
import statsmodels.formula.api as sm

dataset=pd.read_csv('bigData.csv')
p =dataset.head(10)
#print(p)
x=dataset.iloc[:,[1,2,3,4,5]].values
y=dataset.iloc[:,[6]].values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
le = LabelEncoder()
x[:,1]= le.fit_transform(x[:,1])
#print(x[:,1])
x[:,3]= le.fit_transform(x[:,3])
#print(x[:,3])
oneHE = OneHotEncoder(categorical_features=[1,3])

x = oneHE.fit_transform(x).toarray()
x = x[:,2:]
#print(x)

#print(x[:,3])
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
#print(x_train)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
##print(y_test)
#print(y_pred)
from sklearn.metrics import mean_squared_error,r2_score
rms=np.sqrt(mean_squared_error(y_test,y_pred))
print(rms)
r2_score=r2_score(y_test,y_pred)
print(r2_score)





