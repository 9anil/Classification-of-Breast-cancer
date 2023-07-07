# Classification of Breast cancer: The goal is to classify whether the breast cancer is benign or malignant.
# Load the data for analysis and model creation.
import numpy as np
import pandas as pd
dataset=pd.read_csv("data_breastcancer_classification.csv")
#print(dataset.head(20))
# Data cleaning/data preprocessing/data wrangling
dataset['diagnosis']=dataset['diagnosis'].apply({'Benign':0,'Malignant':1}.get)
clean_dataset=dataset.drop(['id'],axis=1)
#print(clean_dataset.head())
# Divide the data in independent(x) and dependent(y)
x=clean_dataset.drop(['diagnosis'],axis=1)
y=clean_dataset[['diagnosis']]
# print(x.head())
# print(y.head())
# Split the data in to training and testing set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
# KNN(K Nearest Neighbour) machine learning model for classification.
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=5)# 5- number of nearest neighbors(K)
knn.fit(x_train,y_train)# Train the machine model for future prediction
predictions=knn.predict(x_test)
print(predictions)
# Confusion matrix for accuracy score in classification
from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(y_test,predictions)
ac=accuracy_score(y_test,predictions)
print(cm)
print(ac)




