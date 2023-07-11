#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#The objective of the notebook is to predict the price of Airbnb hotels in New york city.


# In[1]:


#Load all the required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


# In[ ]:


#This dataset has around 49,000 observations in it with 16 columns and it is a mix between categorical and numeric values


# In[2]:


data = pd.read_csv("C:/Users/user/AB_NYC_2019.csv")
data.head(5)


# In[3]:


data.describe()


# In[4]:


print("the number of features",data.shape[1])
print("the number of entries",data.shape[0])


# In[5]:


data.info()


# In[6]:


#Plotting heatmap of missing values
plt.figure(figsize=(17, 5))
sns.heatmap(data.isnull(), cbar=True, cmap='Set3')
plt.xlabel("Column_Name", size=14, weight="bold")
plt.title("Places of missing values in column",fontweight="bold",size=14)
plt.show()


# In[7]:


#Percentage of missing values in each column
perc=[per for per in data.columns if data[per].isnull().sum()>1]

for per in perc:
    print(per,np.round(data[per].isnull().mean(),3),'% missing values')
    #Its clearly visible that reviews_per_month has maximum number of missing values.


# In[8]:


plt.figure(figsize=(14, 6))
sns.barplot(data=data,x=data.neighbourhood_group,y= data.price, hue=data.room_type, ci=None)


# In[ ]:


""""The above bar plot demonstrates:
Manhattan is the most expensive neighbourhood_group
The price of entire home/apt is more than any other room type.
Bronx is the cheapest."""


# In[9]:


plt.figure(figsize=(15, 6))
sns.scatterplot(x=data.longitude,y=data.latitude,hue=data.neighbourhood_group)


# In[ ]:


#The above plot just decibes the demogrphic view of the entries in the data and 
#also provides a clear view of the neighbourhood_groups.


# In[10]:


data = data[data["price"]<1000]
data.head()


# In[11]:


#Handling outliers by removing entries having price > 1000
price1=data["price"][data["price"]<1000].hist()


# In[12]:


data.duplicated().sum()
data.drop_duplicates(inplace=True)


# In[13]:


data.drop(['name','id', 'host_id','last_review'], axis = 1, inplace = True)
data.fillna({'reviews_per_month' : 0}, inplace = True)
data.isnull().sum()
data.dropna(how = 'any', inplace = True)


# In[14]:


data.drop(['host_name','latitude','longitude','neighbourhood','number_of_reviews','reviews_per_month'], axis = 1, inplace = True)
X = data.iloc[:,[0,1,3,4,5]]
y = data['price']


# In[15]:


X = pd.get_dummies(X, prefix=['neighbourhood_group', 'room_type'], drop_first=True)


# In[16]:


X_train, X_test, y_train, y_test =  train_test_split(X,y,test_size = 0.2, random_state= 0)


# In[17]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)


# In[18]:


reg=LinearRegression()
reg.fit(X_train,y_train)
y_pred=reg.predict(X_test)
#R2 score
from sklearn.metrics import r2_score, mean_squared_error
print("Mean Squared Error: ",r2_score(y_test,y_pred)*100)
print("Root Mean Squared Error: ",np.sqrt(mean_squared_error(y_test,y_pred)))

#Error
error_diff = pd.DataFrame({'Actual Values': np.array(y_test).flatten(), 'Predicted Values': y_pred.flatten()})
print(error_diff.head(5))


# In[ ]:





# In[ ]:





# In[ ]:




