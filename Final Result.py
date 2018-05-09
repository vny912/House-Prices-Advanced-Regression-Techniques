from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df=pd.read_csv('train.csv',sep=',')

x=df[['OverallQual','GrLivArea','YearBuilt','YearRemodAdd','TotalBsmtSF','1stFlrSF','FullBath','TotRmsAbvGrd','GarageCars','GarageArea']]
y=df[['SalePrice']]

##x=df[['OverallQual','GrLivArea','YearBuilt','YearRemodAdd','TotalBsmtSF','1stFlrSF','FullBath','TotRmsAbvGrd','GarageCars','GarageArea']]
##y=df[['SalePrice']]
x_train,x_test,y_train,y_test=train_test_split(x,y)
reg=LinearRegression()
reg.fit(x_train,y_train)
y_predict=reg.predict(x_test)

print('R-squared:',reg.score(x_test,y_test))
zx=reg.score(x_test,y_test)

df1=pd.read_csv('test.csv',sep=',')
x1=df1[['OverallQual','GrLivArea','YearBuilt','YearRemodAdd','TotalBsmtSF','1stFlrSF','FullBath','TotRmsAbvGrd','GarageCars','GarageArea']]
x1.fillna(x1.mean(),inplace=True)
c=x1.shape[0]
a=[]
for i in range(c):
    a1=x1.iloc[i:(i+1),0:11]
    z1=reg.predict(a1)
    a.append(z1[0][0])
    del(a1)
    del(z1)


f=open('Result_%f.csv'%(reg.score(x_test,y_test)),"w")
headers="Id,SalePrice\n"
f.write(headers)
for j in range(c):
    Id=j+1461

    SalePrice=a[j]

    d=str(Id).replace(","," ")+","+str(SalePrice).replace(","," ")

    f.write(d+'\n')

f.close()
