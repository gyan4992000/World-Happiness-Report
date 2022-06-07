#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')


# In[2]:


data=pd.read_csv('https://raw.githubusercontent.com/dsrscientist/DSData/master/happiness_score_dataset.csv')
data.head()


# In[3]:


data.shape


# In[4]:


data.isna().sum()


# In[5]:


data=data.drop(columns=['Country','Region','Happiness Rank'], axis=1)
data


# In[6]:


plt.figure(figsize=(20,15), facecolor='yellow')
plotnumber=1
for column in data:
    if plotnumber<=9:
        ax=plt.subplot(3,3,plotnumber)
        plt.scatter(data[column],data["Happiness Score"])
        plt.xlabel(column,fontsize=10)
        plt.ylabel('Happiness Score', fontsize=10)
    plotnumber+=1
plt.show()


# In[7]:


df_corr=data.corr().abs()
plt.figure(figsize=(15,10))
sns.heatmap(df_corr,annot=True, annot_kws={'size':16})
plt.show()


# In[8]:


#Now, let's split our data into features and label
x=data.drop(columns="Happiness Score", axis=1)
y=data["Happiness Score"]


# In[9]:


x


# In[10]:


y


# In[11]:


scaler=StandardScaler()
x_scaled=scaler.fit_transform(x)


# In[12]:


x=pd.DataFrame(x_scaled, columns=x.columns)


# In[13]:


x


# In[14]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30)


# In[15]:


x_train.shape,x_test.shape,y_train.shape,y_test.shape


# In[16]:


lr=LinearRegression()


# In[17]:


lr.fit(x_train,y_train)


# In[18]:


lr.score(x_test,y_test)


# In[19]:


y_pred=lr.predict(x_test)


# In[20]:


y_pred


# In[21]:


plt.scatter(y_test,y_pred)
plt.xlabel("Actual Happiness Score")
plt.ylabel("Predicted Happiness Score")
plt.title("Actual Vs Predicted Happiness Score")
plt.show()


# In[22]:


mean_absolute_error(y_test,y_pred)


# In[23]:


mean_squared_error(y_test,y_pred)


# In[24]:


np.sqrt(mean_squared_error(y_test,y_pred))


# In[ ]:




