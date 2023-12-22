#!/usr/bin/env python
# coding: utf-8

# # Skills: Programming - Introduction level
# ### Dr. Mario Silic
# ## Group Project - Wallmart Sales Forecasting
# #### Group Members:
# - Collin Arendsen (21-617-204)
# - Gabriel Oget (22-610-893)
# - Luca Drevermann
# - (Roksolana Malyniak)
# - Qishuangshuang Wang


# In[1]:


import pandas as pd
import sklearn as sk
import numpy as np
from pandas.plotting import autocorrelation_plot as auto_corr
import matplotlib.pyplot as plt  
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import seaborn as sns
import math
from datetime import datetime
from datetime import timedelta
import itertools
import statsmodels.api as sm
import statsmodels.tsa.api as smt
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from statsmodels.tsa.seasonal import seasonal_decompose as season
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.linear_model import LinearRegression 
from sklearn import preprocessing
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima_model import ARIMA
from pmdarima.utils import decomposed_plot
from pmdarima.arima import decompose
from pmdarima import auto_arima


# ## 1. Data Cleaning

# ### 1.1 Importing Datasets

# In[2]:


file_path_features = "features.csv"
df_features = pd.read_csv(file_path_features)
df_features.head()


# In[3]:


file_path_stores = "stores.csv"
df_stores = pd.read_csv(file_path_stores)
df_stores.head()


# In[4]:


file_path_train = "train.csv"
df_train = pd.read_csv(file_path_train).drop(columns=['IsHoliday'])
df_train.head()


# In[5]:


# file_path_train = "test.csv"
# df_test = pd.read_csv(file_path_train)
# df_test.head()


# In[6]:


df = df_train.merge(df_features, on=['Store', 'Date'], how='inner').merge(df_stores, on=['Store'], how='inner')
df.head(5)


# ### 1.2 Feature Preprocessing and Cleaning

# In[7]:


df['Weekly_Sales'].min()


# In[8]:


# sales cannot be negative, so we will replace negative values with 0
df.loc[df.Weekly_Sales < 0, 'Weekly_Sales'] = 0


# In[9]:


df.loc[df['IsHoliday']==True, 'Date'].unique() 


# Holidays

# In[10]:


# seperating the holiday column to specific holiday features
holidaydic = {'superbowl' : ['2010-02-12','2011-02-11','2012-02-10'],
              'laborday' : ['2010-09-10','2011-09-09','2012-09-07'],
              'thanksgiving' : ['2010-11-26','2011-11-25'],
              'christmas' : ['2010-12-31','2011-12-30']
              }

for holiday in holidaydic.keys():
    for date in holidaydic[holiday]:
        df.loc[df['Date']==date, holiday] = 1
    df[holiday].fillna(0, inplace=True)
    df[holiday] = df[holiday].astype(bool)


# In[11]:


sns.set(style="whitegrid")

holiday_sales_df = pd.DataFrame({
    'Holiday': ['Superbowl', 'Labor Day', 'Thanksgiving', 'Christmas', 'No Holiday'],
    'Weekly_Sales': [
        df.loc[df.superbowl == True, 'Weekly_Sales'].mean(),
        df.loc[df.laborday == True, 'Weekly_Sales'].mean(),
        df.loc[df.thanksgiving == True, 'Weekly_Sales'].mean(),
        df.loc[df.christmas == True, 'Weekly_Sales'].mean(),
        df.loc[df.IsHoliday == False, 'Weekly_Sales'].mean()
    ]
})

sns.barplot(x='Holiday', y='Weekly_Sales', data=holiday_sales_df)
plt.title('Mean Weekly Sales by Holiday')


# Date

# In[12]:


df['Date'] = pd.to_datetime(df['Date'])


# In[13]:


# creating year, month and week features
df['year'] = df.Date.dt.year
df['month'] = df.Date.dt.month
df['day'] = df.Date.dt.day
df['WeekOfYear'] = df.Date.dt.isocalendar().week


# Anonymized Marketing Features

# In[14]:


df.isna().sum()


# In[15]:


# assuming that a nan means no investment in the respective marketing feature
df[['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']] = df[['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']].fillna(0)


# ### 2. Data Analysis

# #### 2.1 Time Series Analysis

# Trend

# In[16]:


dfdate = df.groupby(['Date']).agg({'Weekly_Sales': 'sum'}).reset_index()

# Create the lineplot and add a trendline
plt.figure(figsize=(12, 6))
sns.lineplot(x=dfdate.index, y='Weekly_Sales', data=dfdate, errorbar=None)
sns.regplot(x=dfdate.index, y='Weekly_Sales', data=dfdate, ci=None, scatter=False, color='red')

# Add labels and a title
plt.xlabel('Date')
plt.ylabel('Total Weekly Sales')
plt.title('Total Weekly Sales Over Time with Trendline')


# In[17]:


# As can be seen in the above plot, there is no significant trend that would need to be corrected


# seasonality

# In[18]:


dfday = df.groupby(['day']).agg({'Weekly_Sales': 'mean'}).reset_index()

# Create the lineplot and add a trendline
plt.figure(figsize=(12, 6))
sns.lineplot(x=dfday.index, y='Weekly_Sales', data=dfday, errorbar=None)

# Add labels and a title
plt.xlabel('Day of Month')
plt.ylabel('Mean Daily Sales')
plt.title('Total Daily Sales over Month')


# In[19]:


dfmonth = df.groupby(['month']).agg({'Weekly_Sales': 'mean'}).reset_index()

# Create the lineplot and add a trendline
plt.figure(figsize=(12, 6))
sns.lineplot(x=dfmonth.index, y='Weekly_Sales', data=dfmonth, errorbar=None)

# Add labels and a title
plt.xlabel('Month')
plt.ylabel('Mean Monthly Sales')
plt.title('Mean Monthly Sales over Year')


# In[20]:


dfweek = df.groupby(['WeekOfYear']).agg({'Weekly_Sales': 'mean'}).reset_index()

# Create the lineplot and add a trendline
plt.figure(figsize=(12, 6))
sns.lineplot(x=dfweek.index, y='Weekly_Sales', data=dfweek, errorbar=None)

# Add labels and a title
plt.xlabel('Month')
plt.ylabel('Mean Monthly Sales')
plt.title('Mean Monthly Sales over Year')


# In[21]:


dfcorrected = df.loc[df.IsHoliday == False] # correcting for holidays, as they are not representative of the general trend

plt.figure(figsize=(16, 10))

# Loop through the selected features and create scatter plots with regression lines
for i, feature in enumerate(['Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']):
    plt.subplot(3, 3, i + 1)
    sns.scatterplot(x=feature, y='Weekly_Sales', data=dfcorrected, alpha=0.1)
    plt.title(f'Scatter Plot of {feature} vs. Weekly Sales')

# Adjust layout
plt.tight_layout()


# In[22]:


df.groupby('Type')['Weekly_Sales'].mean()


# In[23]:


plt.figure(figsize=(6,6))
sns.boxplot(x='Type', y='Size', data=df, showfliers=False)
plt.title('Boxplot of Store Size by Store Type')


# In[24]:


dfdeptmean = df['Weekly_Sales'].groupby(df['Dept']).mean().reset_index()
dfdeptsum = df['Weekly_Sales'].groupby(df['Dept']).sum().reset_index()


fig, axes = plt.subplots(2, 1, figsize=(20, 10), sharey=False)  # Set sharey to False

sns.barplot(x='Dept', y='Weekly_Sales', data=dfdeptmean, ax=axes[0])
axes[0].set_title('Barplot of Weekly Sales by Department (dfdeptmean)')

sns.barplot(x='Dept', y='Weekly_Sales', data=dfdeptsum, ax=axes[1])
axes[1].set_title('Barplot of Weekly Sales by Department (dfdeptsum)')

plt.tight_layout()


# ## 4. Model Training and Comparison

# ### 4.1 Final Preparations

# In[25]:


print(df.info())


# In[26]:


df.index = df['Date']


# In[27]:


df.drop(columns=['Date', 'IsHoliday', 'Size', 'year'], inplace=True)
df = pd.get_dummies(df, columns=['Store', 'Dept', 'month', 'day', 'WeekOfYear', 'Type'])


# In[28]:


# creating different dataframes for each store type
# dfa = df.loc[df['Type']=='A'].reset_index().drop(columns=['Type', 'index'])
# dfb = df.loc[df['Type']=='B'].reset_index().drop(columns=['Type', 'index'])
# dfc = df.loc[df['Type']=='C'].reset_index().drop(columns=['Type', 'index'])


# In[29]:


# splitting the data into train and test sets
split_date = datetime(2012, 4, 26) # six last months for test

train = df[df.index < split_date]
test = df[df.index >= split_date]

X_train = train.drop(columns=['Weekly_Sales'])
y_train = train[['Weekly_Sales']]
X_test = test.drop(columns=['Weekly_Sales'])
y_test = test[['Weekly_Sales']]

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[30]:


def plot_prediction(y_pred_train, y_pred_test, model_name):

    y_train.loc[:, 'y_pred'] = y_pred_train
    y_test.loc[:, 'y_pred'] = y_pred_test

    plt.figure(figsize=(12, 6))
    sns.lineplot(x=y_train.index, y='Weekly_Sales', data=y_train)
    sns.lineplot(x=y_train.index, y='y_pred', data=y_train)
    sns.lineplot(x=y_test.index, y='Weekly_Sales', data=y_test)
    sns.lineplot(x=y_test.index, y='y_pred', data=y_test)

    y_train.drop(columns=['y_pred'], inplace=True)
    y_test.drop(columns=['y_pred'], inplace=True)
    
    plt.xlabel('Date')
    plt.ylabel('Total Weekly Sales')
    plt.title(f'Total Weekly Sales Over Time with {model_name} Predictions')


# ### 4.2 Random Forest

# In[31]:


rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1, max_depth=35, max_features='sqrt', min_samples_split=10)

rf.fit(X_train, y_train)

y_pred_train_rf = rf.predict(X_train)

y_pred_test_rf = rf.predict(X_test)


# In[32]:


plot_prediction(y_pred_train_rf, y_pred_test_rf, 'Random Forest Regressor')


# ### 4.2 Arima

# In[ ]:


arima = auto_arima(train['Weekly_Sales'],
                   seasonal=True,
                   m=12,
                   d=None,
                   start_p=1, start_q=1,
                   max_p=2, max_q=2,
                   max_order=5,     
                   trace=True,
                   error_action='ignore',
                   suppress_warnings=True,
                   stepwise=True,    
                   approximation=True)  


# In[ ]:


y_pred_train_arima = y_train.copy()
y_pred_test_arima = arima.predict(n_periods=len(X_test))


# In[ ]:


plot_prediction(y_pred_train_arima, y_pred_test_arima, 'ARIMA')


# ### 4.3 Exponential Smoothing

# In[ ]:


exsmoo = ExponentialSmoothing(train['Weekly_Sales'], seasonal_periods=20, seasonal='additive', trend='additive', damped=True).fit()


# In[ ]:


y_pred_train_exsmoo = y_train.copy()
y_pred_test_exsmoo = exsmoo.forecast(len(y_test))


# In[ ]:


plot_prediction(y_pred_train_exsmoo, y_pred_test_exsmoo, 'Exponential Smoothing')


# ### 5. Evaluation of Results

# In[ ]:




