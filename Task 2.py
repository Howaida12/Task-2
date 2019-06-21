#!/usr/bin/env python
# coding: utf-8

# # Task 2 : Binary Classification Problem

# In[1]:


import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


# ## Loading Data Sets

# In[2]:


train_data = pd.read_csv("training.csv", sep = ";", header = "infer")
valid_data = pd.read_csv("validation.csv", sep = ";", header = "infer")


# In[3]:


train_data.head()


# In[4]:


valid_data.head()


# ## Preprocessing

# In[5]:


#Checking if there are any missing values
print train_data.isnull().values.any()
print valid_data.isnull().values.any()


# In[6]:


#Checking which columns are already numerical
train_data.dtypes


# In[7]:


valid_data.dtypes


# In[8]:


#A list of the columns that are numerical but exist as objects
num_convert = ["variable2", "variable3", "variable8"]

#Convert those columns to be numerical in the dataframe
for col in num_convert:
    train_data[col] = train_data[col].astype("str")
    train_data[col] = train_data[col].str.replace(",", ".")
    train_data[col] = pd.to_numeric(train_data[col], errors = "coerce")
    
    valid_data[col] = valid_data[col].astype("str")
    valid_data[col] = valid_data[col].str.replace(",", ".")
    valid_data[col] = pd.to_numeric(valid_data[col], errors = "coerce")


# In[9]:


train_data.dtypes


# In[10]:


valid_data.dtypes


# In[11]:


#Setting a list of the categorical columns and another to numerical ones
categoricals = ["variable1", "variable4", "variable5", "variable6", "variable7", 
                "variable9",  "variable10", "variable12", "variable13", "variable18", "classLabel"]
numerics = np.setdiff1d(train_data.columns.tolist(), categoricals)


# In[12]:


#Fill missing data with the mean value of each numerical column
for col in numerics:
    train_data[col] = train_data[col].fillna((train_data[col].mean()))
    valid_data[col] = valid_data[col].fillna((valid_data[col].mean()))


# In[13]:


#Fill missing data with forward fill for train_data, while backfill valid_data as the first row contains NANs
for col in categoricals:
    train_data[col] = train_data[col].fillna(method="ffill")
    valid_data[col] = valid_data[col].fillna(method="backfill")


# In[14]:


#Checking that all missing values are filled
print train_data.isnull().values.any()
print valid_data.isnull().values.any()


# In[15]:


#Add a Target column to contain the same values of classLabel but "yes" = 1 & "no" = 0
train_data['Target'] = train_data['classLabel'].apply(lambda x: 0 if x=='no.' else 1)
valid_data['Target'] = valid_data['classLabel'].apply(lambda x: 0 if x=='no.' else 1)


# In[16]:


#Storing Target column in a separate list then remove them from both train_data and valid_data
train_target = train_data["Target"]
train_data = train_data.drop("classLabel", axis = 1)
train_data = train_data.drop("Target", axis = 1)

valid_target = valid_data["Target"]
valid_data = valid_data.drop("classLabel", axis = 1)
valid_data = valid_data.drop("Target", axis = 1)


# In[17]:


#Remove classLabel from categoricals list as it doesn't exist in the data sets anymore
categoricals.remove("classLabel")
#Create matrices of 0s and 1s in place of the strings in the categorical columns
train_categ = pd.get_dummies(train_data[categoricals].astype(str))
valid_categ = pd.get_dummies(valid_data[categoricals].astype(str))


# In[18]:


#Due to the difference of columns numbers between train_categ and valid_categ
#the missing columns are added but with 0 values
missing_cols = set(train_categ.columns) - set(valid_categ.columns)
for col in missing_cols:
    valid_categ[col] = 0


# In[19]:


#Merge categorical and numerical data
train_merged = pd.merge(train_data[numerics], train_categ, left_index = True, right_index = True, how = "inner")
valid_merged = pd.merge(valid_data[numerics], valid_categ, left_index = True, right_index = True, how = "inner")


# In[20]:


#To make sure that both data sets have their columns arranged in the same order
valid_merged = valid_merged[train_merged.columns]


# ## Logistic Regression Model

# In[21]:


logistic_model = LogisticRegression()
logistic_model.fit(train_merged, train_target)


# In[22]:


valid_predictions = logistic_model.predict(valid_merged)


# In[23]:


logistic_accuracy = accuracy_score(valid_target, valid_predictions)
logistic_precision = precision_score(valid_target, valid_predictions)
logistic_recall = recall_score(valid_target, valid_predictions)
logistic_f1 = f1_score(valid_target, valid_predictions)

print "Logistic Regression Model Accuracy = %f" % logistic_accuracy
print "Logistic Regression Model Precision = %f" % logistic_precision
print "Logistic Regression Model Recall = %f" % logistic_recall
print "Logistic Regression Model F1 = %f" % logistic_f1


# ## Nearest Neighbor Model

# In[24]:


nn_model = KNeighborsClassifier(n_neighbors = 3)
nn_model.fit(train_merged, train_target)


# In[25]:


valid_predictions_2 = nn_model.predict(valid_merged)


# In[26]:


nn_accuracy = accuracy_score(valid_target, valid_predictions_2)
nn_precision = precision_score(valid_target, valid_predictions_2)
nn_recall = recall_score(valid_target, valid_predictions_2)
nn_f1 = f1_score(valid_target, valid_predictions_2)

print "Nearest Neighbor Model Accuracy = %f" % nn_accuracy
print "Nearest Neighbor Model Precision = %f" % nn_precision
print "Nearest Neighbor Model Recall = %f" % nn_recall
print "Nearest Neighbor Model F1 = %f" % nn_f1

