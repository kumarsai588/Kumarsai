#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


df=pd.read_csv(r"C:\Users\admin\Downloads\combined_cycle_power_plant.csv",sep=";")
df.head()


# In[31]:


df.shape


# In[6]:


df.isnull().sum()


# In[8]:


df.drop_duplicates().head()


# In[11]:


a=df.columns.tolist()
a


# In[21]:


#univariate analysis using histplot of continuous variable
for i in a:
    print(df[i].describe())
    sns.histplot(x=i,data=df,color='green')
    plt.show()


# In[13]:


#BIVARIATE ANALYSIS
for i in a:
    sns.scatterplot(x=i,y="energy_output",data=df)
    plt.show()


# In[14]:


#energy_output and temperature has negative linear relationship
#ambigous_pressure and the energy_output has also the positive linear relation
#so it is best for to choose the linear regression model


# In[22]:


#here the linear relationship can also be checked by using the correlation heat map
cor=df.corr()
cor


# In[26]:


#by using the correlation heat map
heat_map=sns.heatmap(cor,annot=True)
heat_map


# In[27]:


#by the correlation heatmap almost all features have the linear relationship is high with kpi that is energy_output


# In[34]:


#seperate tables for dependent features
x=df.drop("energy_output",axis=1)
y=df["energy_output"]


# In[35]:


x.head()


# In[45]:


from sklearn.model_selection import train_test_split


# In[46]:


x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8)


# In[47]:


x_train.shape   #7654 samples for training and 1914 samples for testing purpose that is 20 percent


# In[48]:


x_test.shape


# In[49]:


from sklearn.linear_model import LinearRegression


# In[51]:


lin_algo=LinearRegression()
lin_algo.fit(x_train,y_train)


# In[52]:


lin_algo.intercept_


# In[54]:


lin_algo.coef_


# In[59]:


y_pred=lin_algo.predict(x_test)
y_pred


# In[62]:


from sklearn.metrics import mean_absolute_error  
print(mean_absolute_error(y_test,y_pred))


# In[63]:


#here the mean absolute error is 3.66 so the linear model is good for predicting the energy_output


# In[66]:


np.array(y_test)  #actual values


# In[65]:


y_pred   #predicted values


# In[67]:


#deploying this model using joblib library
import joblib


# In[69]:


joblib.dump(lin_algo,r"C:\Users\admin\OneDrive\Desktop\CCPP_linear_model.pkl")
#saving with .pickle format that is treated as a object


# In[ ]:




