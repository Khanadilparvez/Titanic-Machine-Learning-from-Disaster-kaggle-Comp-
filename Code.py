#importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing Training data and testing data
df=pd.read_csv("train.csv")
df1=pd.read_csv("test.csv")
df.head(5)

#Droping the useless Columns from data set
df.drop(['PassengerId','Fare','Name','Ticket','Cabin'],axis=1,inplace=True)

#taking mean of Age to fill NA
df["Age"]=df["Age"].fillna(29.699)
df["Age"].mean()

#performing Factorization to convert [Gender into 0/1] and Embarked[Q,C,S into 0/1/2]
df['Sex'],_=pd.factorize(df['Sex'])
df['Embarked'],_=pd.factorize(df['Embarked'])

#seprating y=inputs and y=Output 
x=df.drop(["Survived"],axis=1)
y=df["Survived"]

#spliting the test data set to build the model and after building to predict the accuracy....
from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.2,random_state=0)


#For Prediction of Survival [Yes/No] Desicion Tree Classifier is Used.
from sklearn.tree import DecisionTreeClassifier  
classi=DecisionTreeClassifier(criterion="entropy",max_depth=3,random_state=0)  
classi.fit(train_x,train_y)

y_pred=classi.predict(test_x)

#to predict Accuracy using matrix method
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(test_y,y_pred)

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
I get Accuracy of 76% which is now good as Intermediate Stage.....
now i am learning more to increase Accuracy......
i got Public Score=0.77033 in Kaggle Competition
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

#now performing same on test data to predict Survivial of Passengers
PassengerID=df1["PassengerId"]
df1.drop(['PassengerId','Fare','Name','Ticket','Cabin'],axis=1,inplace=True)

df1['Sex'],_=pd.factorize(df1['Sex'])
df1['Embarked'],_=pd.factorize(df1['Embarked'])

df1["Age"].mean()
df1["Age"]=df1["Age"].fillna(30.27)

#Predicing Survival
YPREDICT=classi.predict(df1)

#to conver YPREDICT(INT type) to Dataframe
Y_Pred=pd.DataFrame(YPREDICT.reshape(-1,1))

#merging Passengerid column and Y_pred to form Submission file
Submission=pd.concat([PassengerID,Y_Pred],axis=1)

#saving csv file
