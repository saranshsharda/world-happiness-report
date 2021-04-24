#!/usr/bin/env python
# coding: utf-8

# In[45]:


import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')

import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")


# In[46]:


df = pd.read_csv(r"C:\Users\HP\Desktop\happiness.csv",engine='python')


# In[47]:


df.head()


# In[48]:


df.shape


# In[49]:


df=df.drop(['Region', 'Standard Error', 'Dystopia Residual'], axis=1)


# In[50]:


df.head()


# In[51]:


df.rename(columns = {"Happiness Rank": "Rank", 'Happiness Score':'Score',
                         'Economy (GDP per Capita)':'GDP', 
                         'Health (Life Expectancy)':'Life expectancy',                          
                'Trust (Government Corruption)':'Corruption'},  
          inplace = True) 


# In[52]:


corr1=df.corr()
sns.heatmap(corr1, cmap="icefire")


# In[53]:


plt.figure(figsize=(17, 6))
g=sns.barplot(x ='Country', y ='Rank', data = df, palette ='Paired') 
plt.xticks(rotation=90)


# In[54]:


sns.lmplot(x='Rank', y='Life expectancy', data=df, logistic=False)


# In[55]:


a=df['Rank']

sns.kdeplot (a, label="2015 Happiness Rank")


# In[56]:


f, axes = plt.subplots(3, 2, figsize=(13, 8), sharex=False)
sns.barplot(df.Country.head(5) , df.Rank , data = df, palette ='Set1', ax=axes[0, 0])


# In[60]:


pd.set_option('display.max_rows', None)
c=df.sort_values(by=['Country'])
c


# In[61]:


##2015
x=df['Country'].values
y=df['Family'].values
fig, ax = plt.subplots(figsize =(10, 24))   
colors='salmon','navy','forestgreen', 'goldenrod'

ax.barh(x, y, color=colors, edgecolor=(0,0,0)) 
ax.set_title('2015 Family Support by Country') 
plt.show()


# In[64]:


fig, axes = plt.subplots(3, 2, figsize=(10,8), sharex=False)
df.plot(x='GDP', y=['Corruption'], ax=axes[0, 0], color='darkorange')


# In[65]:


df.head()


# In[70]:


col=df[['GDP','Life expectancy','Freedom','Generosity','Corruption','Score']].corr()
plt.figure(figsize=(10,7))
sns.heatmap(col,annot=True,cmap="Greens")
plt.title('Correlattion of 2015 Dataset')
plt.show()


# In[73]:


fig, axes = plt.subplots(nrows=3, ncols=2,constrained_layout=True,figsize=(10,10))

sns.barplot(x='Life expectancy' ,y='Country',
                        data=df.nlargest(10,'Life expectancy'),
                        ax=axes[0,0],palette="rocket")
sns.barplot(x='Score' ,y='Country',
                        data=df.nlargest(10,'Score'),
                        ax=axes[1,1],palette="rocket")
sns.barplot(x='Generosity' ,y='Country',
                        data=df.nlargest(10,'Generosity'),
                        ax=axes[1,0],palette="rocket")
sns.barplot(x='Freedom' ,y='Country',
                        data=df.nlargest(10,'Freedom'),
                        ax=axes[2,1],palette="rocket")
sns.barplot(x='Corruption' ,y='Country',
                        data=df.nlargest(10,'Corruption'),
                        ax=axes[2,0],palette="rocket")


# In[106]:


y=df['Score']
x=df[['Family', 'Life expectancy','Freedom', 'Generosity',
           'Corruption']]


# In[108]:


x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2, shuffle=True, random_state=13579)


# In[109]:


lr = LinearRegression()
lr.fit(x_train,y_train)
    
    
 #prediction for train test and test test
y_pred_train = lr.predict(x_train)
y_pred_test = lr.predict(x_test)
    
#Training and Testing MSE and RMSE
train_mse = np.mean((y_train - lr.predict(x_train)) ** 2)
test_mse = np.mean((y_test - lr.predict(x_test))** 2)
print('   Linear   Regression    ')
print("Train MSE is",train_mse)
print("Test MSE is",test_mse)
print("Train RMSE is:",np.sqrt(train_mse))
print("Test RMSE is",np.sqrt(test_mse))
        
#r2 Score value
r2 = r2_score(y_test,y_pred_test)
print("r2 score is",r2)
print('   Linear   Regression    ')
print()
    
#Plotting Residual Plot
residual=y_train-y_pred_train
residual=residual.values.reshape(len(residual),1)
plt.scatter(y_train,residual,c = "red")
plt.xlabel("residual")
plt.ylabel("y_test")
plt.axhline(y = 0)
    
# Checming Normality Condition
import scipy as sp
fig, ax = plt.subplots(figsize=(6,3))
_, (__, ___, r) = sp.stats.probplot(residual.reshape((len(residual),)), plot=ax, fit=True)


# In[111]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


# In[133]:


from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import sklearn.metrics as metrics
import warnings
warnings.filterwarnings("ignore")
np.random.seed(42)


# In[134]:


lr1 = LogisticRegression()


# In[136]:


lr.fit(x_train,y_train)


# In[137]:


cv_score= cross_val_score(lr,x_train,y_train, cv=5)
np.mean(cv_score)


# In[138]:


y_pred = lr.predict(x_test)
lr.score(x_test,y_test)


# In[ ]:





# In[ ]:




