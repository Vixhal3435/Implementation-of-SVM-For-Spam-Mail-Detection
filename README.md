# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Data Collection

Obtain a labeled dataset with features such as email content, subject lines, word frequencies, and labels (Spam or Ham).
Example datasets include the Enron spam dataset or any public spam dataset like SMS Spam Collection.

2.Data Preprocessing

Clean the text data: Remove special characters, stop words, and normalize the text by converting it into lowercase.
Extract meaningful features using text vectorization techniques such as TF-IDF (Term Frequency-Inverse Document Frequency) or Count Vectorizer.

3.Feature Extraction

Convert the email text into numerical data suitable for machine learning models. This is usually done by transforming the raw text data into a vector form (e.g., using a Bag-of-Words model or TF-IDF vectorizer).

4.Split Data

Divide the dataset into training and testing sets for model evaluation.

5.Train the SVM Model

Use the SVM classifier from scikit-learn. It will try to create a hyperplane that best separates spam and non-spam emails. Select the kernel (usually linear or radial basis function (RBF)).

6.Model Evaluation

Evaluate the model using performance metrics such as accuracy, precision, recall, F1-score, and confusion matrix.

7.Spam Prediction

Use the trained model to predict whether new emails are spam or not.

## Program:
```
import chardet
file=(r'C:\Users\admin\Downloads\spam.csv')
with open(file,'rb')as rawdata:
    result=chardet.detect(rawdata.read(100000))
print(result)
import pandas as pd
data=pd.read_csv(r'C:\Users\admin\Downloads\spam.csv',encoding='Windows-1252')
print(data.head())
print(data.info())
data.isnull().sum()
x=data["v1"].values
y=data["v2"].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
print(y_pred)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
print(accuracy)
```
Program to implement the SVM For Spam Mail Detection..

Developed by: VISHAL.V

RegisterNumber: 24900179 

## Output:

![OUTPUT FOR EX 11](https://github.com/user-attachments/assets/ed8d9302-a48d-47d4-8b87-08b53e0631bb)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
