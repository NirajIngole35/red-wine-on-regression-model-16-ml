

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


ds=pd.read_csv(r"C:\Users\HP\Desktop\INTERNSHIP 5 MARCH\ml project\red wine on regression model  16 ml\winequality-red.csv")
print(ds)
print(ds.head(4))
print(ds.tail(4))
print(ds.shape)
print(ds.info)
print(ds.describe)

#find x and y

x=ds.iloc[:,:4].values
y=ds.iloc[:,-1].values
yb=y.reshape(len(y),1)
print(x)
print(y)

#graph 
import matplotlib.pyplot as plt
plt.boxplot(x,data=ds)
plt.show()
#train_test

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)
#train_test

x_trainb,x_testb,y_trainb,y_testb=train_test_split(x,yb,test_size=0.2,random_state=1)



from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn .ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
model_1=LinearRegression()
model_1.fit(x_train,y_train)

model_2=DecisionTreeRegressor()
model_2.fit(x_train,y_train)


model_3=SVR()
model_3.fit(x_train,y_train)


model_4=RandomForestRegressor()
model_4.fit(x_train,y_train)


pol=PolynomialFeatures(degree=4)
x_po=pol.fit_transform(x_train)
model_5=LinearRegression()
model_5.fit(x_trainb,y_trainb)

from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
sc2=StandardScaler()
x_trainb=sc1.fit_transform(x_trainb)
y_trainb=sc2.fit_transform(y_trainb)

#predict

model_4_pre=model_4.predict(x_test)
model_3_pre=model_3.predict(x_test)
model_2_pre=model_2.predict(x_test)


#result

from sklearn.metrics import r2_score
print(r2_score(y_test,model_2_pre)*100)
print(r2_score(y_test,model_3_pre)*100)
print(r2_score(y_test,model_4_pre)*100)