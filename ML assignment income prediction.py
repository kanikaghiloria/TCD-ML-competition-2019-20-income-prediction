#!/usr/bin/env python
# coding: utf-8

# In[1]:

import pandas as pd
from pandas import DataFrame
import numpy as np
from sklearn import preprocessing, model_selection, linear_model
import sklearn
import matplotlib.pyplot as plt
from sklearn.feature_extraction import DictVectorizer
from category_encoders import target_encoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# In[2]:

#Reading training dataset
dataset = pd.read_csv("tcd ml 2019-20 income prediction training (with labels).csv")
#print (dataset.head())

# In[3]:

#Selecting features
dataset=dataset.drop(['Wears Glasses' , 'Hair Color'], axis = 1)
#dataset.to_csv('dataset.csv', index=False)
#print (dataset.head())
#print (dataset.shape)

# In[4]:

#Reading prediction dataset
pred_dataset = pd.read_csv("tcd ml 2019-20 income prediction test (without labels).csv")
pred_dataset= pred_dataset.drop(['Wears Glasses' , 'Hair Color'], axis = 1)
#print (pred_dataset.head())

# In[5]:

#Removing outliers
dataset = dataset[dataset['Income in EUR'] < 3000000]
dataset = dataset[dataset['Income in EUR'] >= 0]

# In[6]:

#Managing NAN values

#dataset.dropna(inplace=True)
dataset['Year of Record'].fillna(method='pad', inplace=True)
dataset['Gender'].fillna('unknown', inplace=True)
dataset['Profession'].fillna('#N/A', inplace=True)
dataset['Age'].fillna(dataset['Age'].mean(), inplace=True)
dataset['University Degree'].fillna('No', inplace=True)
#dataset['Hair Color'].fillna('Unknown', inplace=True)
#print (dataset.head())
#dataset.to_csv('dataset_without outliers.csv', index=False)


# In[7]:

#Processing the feature set
dataset['Gender'].replace('0', 'unknown', inplace=True)
dataset['University Degree'].replace('0', 'No', inplace=True)
#dataset['Hair Color'].replace('0', 'Unknown', inplace=True)

#print (dataset.head())
#dataset.to_csv('updated_dataset.csv', index=False)
#print (dataset.dtypes)


# In[8]:

#Processing the features in prediction data
pred_dataset['Year of Record'].fillna(method='pad', inplace=True)
pred_dataset['Gender'].fillna('unknown', inplace=True)
pred_dataset['Profession'].fillna('#N/A', inplace=True)
pred_dataset['Age'].fillna(pred_dataset['Age'].mean(), inplace=True)
pred_dataset['University Degree'].fillna('No', inplace=True)
pred_dataset['Body Height [cm]'].fillna(pred_dataset['Body Height [cm]'].mean(), inplace=True)

pred_dataset['Gender'].replace('0', 'unknown', inplace=True)
pred_dataset['University Degree'].replace('0', 'No', inplace=True)
#print (pred_dataset.head())


# In[9]:

#Plotting the features

#fig, ax = plt.subplots()
#ax.scatter(dataset['Year of Record'], dataset['Income in EUR'])
#ax.set_xlabel('Year of Record')
#ax.set_ylabel('Income in EUR')


# In[10]:


#fig, ax = plt.subplots()
#ax.scatter(dataset['Gender'], dataset['Income in EUR'])
#ax.set_xlabel('Gender')
#ax.set_ylabel('Income in EUR')


# In[11]:


#fig, ax = plt.subplots()
#ax.scatter(dataset['Age'], dataset['Income in EUR'])
#ax.set_xlabel('Age')
#ax.set_ylabel('Income in EUR')


# In[12]:


#fig, ax = plt.subplots()
#ax.scatter(dataset['Country'], dataset['Income in EUR'])
#ax.set_xlabel('Country')
#ax.set_ylabel('Income in EUR')


# In[13]:


#fig, ax = plt.subplots()
#ax.scatter(dataset['Size of City'], dataset['Income in EUR'])
#ax.set_xlabel('Size of City')
#ax.set_ylabel('Income in EUR')


# In[14]:


#fig, ax = plt.subplots()
#ax.scatter(dataset['Profession'], dataset['Income in EUR'])
#ax.set_xlabel('Profession')
#ax.set_ylabel('Income in EUR')


# In[15]:


#fig, ax = plt.subplots()
#ax.scatter(dataset['University Degree'], dataset['Income in EUR'])
#ax.set_xlabel('University Degree')
#ax.set_ylabel('Income in EUR')


# In[16]:


#fig, ax = plt.subplots()
#ax.scatter(dataset['Wears Glasses'], dataset['Income in EUR'])
#ax.set_xlabel('Wears Glasses')
#ax.set_ylabel('Income in EUR')


# In[17]:


#fig, ax = plt.subplots()
#ax.scatter(dataset['Hair Color'], dataset['Income in EUR'])
#ax.set_xlabel('Hair Color')
#ax.set_ylabel('Income in EUR')


# In[18]:


#fig, ax = plt.subplots()
#ax.scatter(dataset['Body Height [cm]'], dataset['Income in EUR'])
#ax.set_xlabel('Body Height [cm]')
#ax.set_ylabel('Income in EUR')


# In[19]:

#Handling categorical features using label Encoder

#le= preprocessing.LabelEncoder()
#dataset['Gender'] = le.fit_transform(dataset['Gender'])
#dataset['Country'] = le.fit_transform(dataset['Country'])
#dataset['Profession'] = le.fit_transform(dataset['Profession'])
#dataset['University Degree'] = le.fit_transform(dataset['University Degree'])
#dataset['Hair Color'] = le.fit_transform(dataset['Hair Color'])

#print (dataset.head())

#dataset.to_csv('updated_dataset_encoder.csv', index=False)


# In[20]:


y = np.array(dataset['Income in EUR']) # training data
dataset=dataset.drop(['Instance','Income in EUR'], axis=1) # training data features
X_pred = pred_dataset.drop(['Instance','Income'], axis=1) #prediction data feature
#X = np.array(dataset.drop(['Instance', 'Income in EUR' ], 1))
#print ("y : ", y)
#print ("X : ", X)
#print (dataset.head())

#Splitting the data into training and prediction sets
X_train, X_test, y_train, y_test = model_selection.train_test_split(dataset,y,test_size = 0.2)
#print ("X_train : ", X_train.head())
#print ("X_test : ", X_test.head())
#print ("y_train : ", y_train)
#print ("y_test : ", y_test)

# In[21]:

#Handling categorical features using target encoder
te = target_encoder.TargetEncoder(return_df=False)
X = te.fit_transform(dataset, y)
X_pred = te.transform(X_pred)

te_test = target_encoder.TargetEncoder(return_df=False)
X_train = te_test.fit_transform(X_train, y_train)
X_test = te_test.transform(X_test)
#Handling categorical features using DictVectorizer
#dv_X = DictVectorizer(sparse=False)
#X = dv_X.fit_transform(dataset.to_dict(orient='record'))
#X_pred = dv_X.transform(pred_dataset.to_dict(orient='record'))
#print ("X : ", X)
#print ("X_pred : ", X_pred)

#dv_X_test = DictVectorizer(sparse=False)
#X_train = dv_X_test.fit_transform(X_train.to_dict(orient='record'))
#X_test = dv_X_test.transform(X_test.to_dict(orient='record'))
#print ("X_train : ", X_train)
#print ("X_test : ", X_test)
#print ("y_train : ", y_train)
#print ("y_test : ", y_test)
#print (X_train.shape)

# In[24]:

#Scaling the feature set
#print ("Scaling")
scaler = preprocessing.StandardScaler(copy=False)
X = scaler.fit_transform(X)
X_pred = scaler.transform(X_pred)
#X = preprocessing.scale(X, copy=False)
#X_pred = preprocessing.scale(X_pred, copy=False)
#print ("X : ", X)
#print ("X_pred : ", X_pred)

scaler_test = preprocessing.StandardScaler(copy=False)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
#X_train = preprocessing.scale(X_train, copy=False)
#X_test = preprocessing.scale(X_test, copy=False)
#np.savetxt("X_scale.csv", X, delimiter=",")
#print ("X_train : ", X_train)
#print ("X_test : ", X_test)


# In[ ]:


#print (X.shape)
#print(y.shape)


# In[ ]:

#Training the data

clf = GradientBoostingRegressor(learning_rate = 0.09, n_estimators = 1200)
#clf =  linear_model.Lasso()
#clf =  linear_model.Ridge()
#clf = RandomForestRegressor(n_estimators = 500)
clf.fit(X_train, y_train)
y_pred_test = clf.predict(X_test)
#print ("y_pred_test : " , y_pred_test)

#Calculating the expected score
mse = mean_squared_error(y_test,y_pred_test) 
rmse = np.sqrt(mse)
print ('{0:f}'.format(rmse))

classifier = GradientBoostingRegressor(learning_rate = 0.09, n_estimators = 1200)
#classifier =  linear_model.Lasso()
#classifier =  linear_model.Ridge()
#classifier = RandomForestRegressor(n_estimators = 500)
classifier.fit(X,y)


# In[ ]:

#predicting the outcome
y_pred = classifier.predict(X_pred)
print ("y_pred: ", y_pred)


# In[ ]:

#printing the outcome to the csv file
pred_dataset['Income'] = y_pred
pred_dataset = pred_dataset.drop(['Year of Record', 'Gender','Age','Country', 'Size of City', 'Profession', 'University Degree','Body Height [cm]'], axis=1)
#print (pred_dataset.head())
pred_dataset.to_csv('tcd ml 2019-20 income prediction submission file.csv', index=False)

