# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import the required libraries.

2.Upload and read the dataset.

3.Check for any null values using the isnull() function.

4.From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.

5.Find the accuracy of the model and predict the required values by importing the required module from sklearn.
```
## Program:
```
/*
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: SWETHA S V
RegisterNumber:212224230285
*/
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier,plot_tree
data=pd.read_csv("C:\\Users\\admin\\Downloads\\Employee (1).csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()    #no departments and no left
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
​
​
plt.figure(figsize=(8,6))
plot_tree(dt,feature_names=x.columns,class_names=['salary','left'],filled=True)
plt.show()

```

## Output:

### DATA HEAD
![image](https://github.com/user-attachments/assets/a537fccb-43f7-4b8d-b673-0a238a0d6e47)
### info
![image](https://github.com/user-attachments/assets/83994d90-02fe-46b4-b5b8-43007e53137c)
### null set
![image](https://github.com/user-attachments/assets/dc6812d1-89d7-4f4e-9904-171d45936ef1)
### VALUES COUNT IN THE LEFT COLUMN
![image](https://github.com/user-attachments/assets/b9ab09a6-d970-4168-888a-5e7dba2eda3e)
### DATASET TRANSFORMED HEAD
![image](https://github.com/user-attachments/assets/9b1fa7be-8174-43bc-8184-fb1269984ad8)
### X.HEAD
![image](https://github.com/user-attachments/assets/d656bac7-607b-4211-8a3c-7e4a61e0fee0)
### ACCURACY
![image](https://github.com/user-attachments/assets/e8d19014-e418-4056-94c7-8f8ad37b4fb8)
### DATA PREDICTION
![Screenshot 2025-05-03 081710](https://github.com/user-attachments/assets/77afc25a-4f96-4e30-8661-dd7beb6d0fc6)
![image](https://github.com/user-attachments/assets/cffa2c95-9bf7-478c-b7bc-56e765d37613)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
