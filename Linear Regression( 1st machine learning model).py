import numpy as np
import pandas as pd
import matplotlib.pyplot as mlt 
import sklearn 
from sklearn import linear_model , datasets 
from sklearn.metrics import mean_squared_error


# importing important modules and libraries .
# mean_sqaured_error for calculating the error between predicted and actual values

diabetes = datasets.load_diabetes()

print(diabetes.data.shape)
# shape tells the length of the data .
# rows will tell about the number of people whose information was taken .
# the column wil tell what data was taken . like( age  , sex , bmi , blood pressure,
# blood sugar level .)


print(diabetes.DESCR)
# the whole description about the dataset of diabetic patient . 

print(diabetes.feature_names)
# the features of the diabetes database . 

print(len(diabetes.feature_names))
# number of features .

# at first lets take just only one features .
# for that . just convert the dataset to column oriented and take any feature .
diabetes_feature  = diabetes.data[: , np.newaxis , 0]
# diabetes_X = diabetes.data[: , np.newaxis , 0 ]
# here out of 10 features , we are just taking 0th , the first label .
# you can select the feature you want . 0 ,1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9

print(diabetes_feature)
# this will show the data of the dibetes patient . 

diabetes_target = diabetes.target
print(diabetes_target)
print(len(diabetes_target))
# this dataset will tell the actual diabetes level of the diabetes patients .
# the corrresponding dataset of diabetes_target tells the diabetes level of 
# diabetes_feature .
# diabetes_target[0] will tell the diabetes level of diabetes.data[0]


# lets create datasets for training data and testing the data .
# the data have 442 people . so lets choose 400 people data for training .
# and rest 42 for testing .

# X refers to features and Y refers to labels . 
# features are the inputs  , which tells that this person has this symptoms 
# or the situations on which diabetes depends .
# and labels is the diabetes level or it tells that the person is diabetic 
# or not 

# selecting first 400 people for training
diabetes_features_train = diabetes_feature[:400]
# diabetes_X_train = diabetes_feature[:400]
diabetes_labels_train   =  diabetes.target[:400]
# diabetes_Y_train  = diabetes.target[:400]

# selecting rest 42 for testing
diabetes_features_test = diabetes_feature[400:] 
# diabettes_X_test = diabetes_feature[400:]
diabetes_labels_test = diabetes.target[400:]
# diabetes_Y_test = diabetes.target[:400]

print(diabetes_features_train)
print(diabetes_labels_train)
print("\n\n")
print(diabetes_features_test)
print(diabetes_labels_test)

# create a model 

model = linear_model.LinearRegression()

# fitting the data or say obtaining the line for regression  .
model.fit(diabetes_features_train , diabetes_labels_train)

diabetes_labels_predicted = model.predict(diabetes_features_test)

for i in range(len(diabetes_labels_predicted)):
    print("Actual -:" , diabetes_labels_test[i] , " Predicted " , diabetes_labels_predicted[i])

print("\n\n")
print("Mean squared error is -:" , mean_squared_error(diabetes_labels_predicted , diabetes_labels_test))
# this will show the mean squared differernce between predicted and actual values

"""
def error(list1 , list2) :
    c= 0 
    for i in range(len(list2)):
        c+= (list1[i] - list2[i])**2
    return c/len(list1)

print(error(diabetes_labels_predicted , diabetes_labels_test))

# this is mean_squared_error and this function will work the same . 
# just remove the multi line string . 

"""

print(model.coef_)
# this gives the slope of the line .

print(model.intercept_)
# this will give the intercept of the line which is predicted .

# now to plot the graph . 


mlt.scatter(diabetes_features_test , diabetes_labels_test  , s = 40 , color = "blue")
# here we have taken the testing data to see position of line from different points.
# is the line good or not .

mlt.plot(diabetes_features_test , diabetes_labels_predicted , color = "red" )
# plotting the line , this will be done by taking the testing data and the 
# predicted labels . which is actual line .

# after running the program , pan the graph to (0 , 0 ) , make sure your origin
# is on ( 0 , 0)  an see the intercept of the graph .
# if its same , then you have done all correct and predicted nicely .

mlt.show()

# for using all the features , just remove the slicing in the line 32
# diabetes_features = diabetes.data 

# and also remove  the plotting lines , because you cannot plot multiple lables and 
# features .

# just copy the whole code or download it and enjoy the program 😎🥰
