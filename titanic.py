import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression,LinearRegression

csv_data=pd.read_csv(r'c:\Users\huang\Downloads\train.csv',
                usecols=['Survived','Pclass','Sex','Age','SibSp','Parch','Fare'])
csv_data=csv_data.replace(['male','female'],[1,0])
csv_data=csv_data.fillna(csv_data.mean())
#csv_data=np.array(csv_data)
x=csv_data[['Pclass','Sex','Age','SibSp','Parch','Fare']]
y=csv_data['Survived']

x_train=np.array(x)[:-100]
y_train=np.array(y)[:-100]

x_test=np.array(x)[-100:]
y_test=np.array(y)[-100:]

linreg=LinearRegression()
linreg.fit(x_train,y_train)

logreg=LogisticRegression(C=10)
#logreg.fit(x_train,y_train)
logreg.fit(x,y)

z=logreg.predict(x_test)
print('log:',logreg.score(x_test,y_test))
print('linear:',linreg.score(x_test,y_test))

all_data=pd.read_csv(r'C:\Users\huang\Downloads\test.csv',
                      usecols=['PassengerId','Pclass','Sex','Age','SibSp','Parch','Fare'])
test_data=all_data[['Pclass','Sex','Age','SibSp','Parch','Fare']]
test_data=test_data.replace(['male','female'],[1,0])
test_data=test_data.fillna(test_data.mean())
print(type(logreg.predict(test_data)))
csv_in=all_data[['PassengerId']]
csv_in['Survived']=pd.Series(logreg.predict(test_data))
csv_in.to_csv(r'C:\Users\huang\Desktop\titanic2.csv',index=False)
