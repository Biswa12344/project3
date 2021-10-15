from sklearn import linear_model
import pandas as pandas
import numpy as np
import pickle
from sklearn.model_selection import train_test_split

df= pandas.read_csv('car.csv')
print(df)
X =df[['Weight','Volume']]
# x1=df['Weight']
# x2=df['Volume']
y=df['CO2']
# regr= linear_model.LinearRegression()
# regr.fit(X,y)
# print(regr.coef_)
# x1_mean= np.mean(x1)
# x2_mean=np.mean(x2)
# y_mean= np.mean(y)
# x1_len= len(x1)
# num1=0
# num2=0
# den1=0
# den2=0
# for i in range(x1_len):
#     num1+= (x1[i]-x1_mean)*(y[i]-y_mean)
#     den1+=(x1[i]-x1_mean)**2
# m1=num1/den1
#
# for i in range(x1_len):
#     num2+= (x2[i]-x2_mean)*(y[i]-y_mean)
#     den2+=(x2[i]-x2_mean)**2
# m2= num2/den2
#
# print('m:',m1)
# print('m2:',m2)
#
# c= y_mean-((-0.00434464*x1_mean)+(0.00485082*x2_mean))
# print(c)
# # #predictedCO2 = regr.predict([[2300, 1300]])
# #
# #
# y_p= (-0.00434464*2300)+(0.00485082*1300)+96.55075159076922
# print('y is :',y_p)

# 92.86414559076923
#
#
#
#
# from sklearn import linear_model
#from sklearn.model_selection import train_test_split

#csv_reader= csv.reader('car.csv')
# df= pandas.read_csv('cars1.csv')
# #csv_reader= df.to_csv('car.csv')
# #print(df)
# #new_header = df. iloc[0]
# #df.columns=new_header
#
# #print(df)
#
# X =df[['Weight','Volume']]
# #x3= df.iloc[:,2:4]
# #print(x3)
# y =df['CO2']
# #print(y)
# #print(x)
#
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
regr = linear_model.LinearRegression()
#regr.fit(X_train, y_train)
regr.fit(X,y)
pickle.dump(regr,open('model.pkl','wb'))

#model=pickle.load(open('model.pkl','rb'))

#predictedCO2 = regr.predict(X_test)
# predictedCO2 = regr.predict([[2300,1300]])
#
#
# print('predicted co2 value is in test 5:',predictedCO2)
