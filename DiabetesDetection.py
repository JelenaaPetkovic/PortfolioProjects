""""
In this project we implement the kNN algorithm on the Pima Indians Diabetes dataset in order to predict whether a patient is likely to have a diabetes.
Skills used: Understanding the data and dealing with missing values, Normalization of the data, Implementation of the kNN algorithm, Applying the cross-validation to infer accuracy

"""

import os 
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import math
import random

# Reading the data
os.chdir("C:\\Users\\User\\Desktop\\PRML")
data=pd.read_excel('Diabetes_data2.xlsx')
data.head(10)
print(data.shape)

#Dealing with missing values
data_copy = data.copy()
data_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']]=data_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)
print(data_copy.isnull().sum())
data_copy.head()

sns.boxplot(data=data[['Glucose', 'BloodPressure', 'SkinThickness','Insulin','BMI']], orient = "h", palette = "Blues_d") 
plt.show() 

data_copy['Glucose'].fillna(data_copy['Glucose'].mean(), inplace = True)
data_copy['BloodPressure'].fillna(data_copy['BloodPressure'].median(), inplace = True) 
data_copy['SkinThickness'].fillna(data_copy['SkinThickness'].median(), inplace = True)
data_copy['Insulin'].fillna(data_copy['Insulin'].median(), inplace = True)
data_copy['BMI'].fillna(data_copy['BMI'].median(), inplace = True)
data_copy.isnull().sum() 

# Z-Score Normalization
def z_score(data):
    data_std=data.copy()
    
    for column in data_std.columns:
        if column != 'Outcome':    
            data_std[column]= ((data_std[column]-data_std[column].mean()) / data_std[column].std())
        
    return data_std

data_stand = z_score(data_copy)
data_stand

#Considering correlation between the features
plt.figure(figsize=(12,10)) 
p=sns.heatmap(data_stand.corr(), annot=True,cmap ='RdYlGn')  
plt.savefig('Heatmap')

#kNN algorithm
data_new=data_stand.values.tolist()  
print(data_new)

def euclidean_distance(row1, row2): 
    distance = 0
    for i in range(len(row1)-1):   
        distance += (row1[i] - row2[i])**2 
    return math.sqrt(distance)
  
def get_neighbors(train, test_row, num_neighbors):
    distances=[]                               
    for train_row in train:                    
        dist=euclidean_distance(test_row, train_row)
        if dist>0:                                           
            distances.append((train_row, dist))           
    
    distances.sort(key=lambda tup: tup[1])          
    neighbors=[]                        
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])   
    return neighbors
  
#make a classification prediction with the neighbors
def predict_classification(train, test_row, num_neighbors):  
    neighbors=get_neighbors(train, test_row, num_neighbors) 
    output_values=[row[-1] for row in neighbors]   
    prediction=max(set(output_values), key=output_values.count) 
    return prediction 
  
def k_nearest_neighbors(train, test, num_neighbors):
    predictions = list()
    for row in test:
        output = predict_classification(train, row, num_neighbors)
        predictions.append(output)
    return(predictions)

# Cross-Validation
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for _ in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = random.randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split
  
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0

# evaluate an algorithm using a Cross-Validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores
  
# Applying the 10-fold Cross-Validation with 27 nearest neighbours
scores=evaluate_algorithm(data_new, k_nearest_neighbors, 10, 27)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
  

