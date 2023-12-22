import streamlit as st
from joblib import load
import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt    
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

# load the dataset that was created in the previous step
df = pd.read_csv('final_df.csv')

# Making some final adjustments to the dataset 
# (especially taking the dummy variables only now, as saving a csv file with them takes a lot of time)
df['Date'] = pd.to_datetime(df['Date'])
df.set_index(['Date'], inplace=True)
df.drop(columns=['IsHoliday', 'Size', 'year', 'day'], inplace=True)
df = pd.get_dummies(df, columns=['Store', 'Dept', 'month', 'WeekOfYear'])

# creating different dataframes for each store type
dfa = df.loc[df['Type']=='A'].drop(columns=['Type'])
dfb = df.loc[df['Type']=='B'].drop(columns=['Type'])
dfc = df.loc[df['Type']=='C'].drop(columns=['Type'])

dflist = [dfa, dfb, dfc]
typedic = {'A': 0, 'B': 1, 'C': 2}

# Split the data into training and testing sets
split_date = datetime(2011, 10, 1)

trainlist = [df[df.index < split_date] for df in dflist]
testlist = [df[df.index >= split_date] for df in dflist]

# Create a function to make predictions, based on the store type and model
def make_prediction(type, model):
    k = typedic[type]
    
    df = dflist[k]
    train = trainlist[k]
    test = testlist[k]
    
    X_train = train.drop(columns=['Weekly_Sales'])
    y_train = train['Weekly_Sales']
    X_test = test.drop(columns=['Weekly_Sales'])
    y_test = test['Weekly_Sales']

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if model == 'Linear Regression':
        param_grid = {
            'fit_intercept': [True, False],
            'alpha': np.logspace(0, 10, 20)
        }
        
        lr = Ridge()
        
        lr_cv = GridSearchCV(estimator=lr, param_grid=param_grid, cv=3, verbose=2, n_jobs=-1)
        
        lr_cv.fit(X_train_scaled, y_train)
        
        print("Best parameters found: ", lr_cv.best_params_)
        
        y_pred_test = lr_cv.predict(X_test_scaled)

    elif model == 'Random Forest Regressor':
        param_grid = {
            'n_estimators': [20, 50],
            'max_depth': [10, 40, 60],
            'max_features': ['sqrt', 'log2']
        }

        rf = RandomForestRegressor()

        rf_cv = RandomizedSearchCV(estimator=rf, param_distributions=param_grid, n_iter=5, cv=3, verbose=2, random_state=42, n_jobs=-1)

        rf_cv.fit(X_train_scaled, y_train)

        print("Best parameters found: ", rf_cv.best_params_)
            
        y_pred_test = rf_cv.predict(X_test_scaled)        
    
    plotdf = pd.DataFrame({
        'Date': test.index,
        'test': test['Weekly_Sales'],
        'pred': y_pred_test
    })
    
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=train.index, y='Weekly_Sales', data=train)
    sns.lineplot(x=plotdf['Date'], y='test', data=plotdf)
    sns.lineplot(x=plotdf['Date'], y='pred', data=plotdf)

    plt.legend(labels=['Train', 'Train Variance', 'Test', 'Test Variance', 'Predictions', 'Prediction Variance'])
    plt.xlabel('Date')
    plt.ylabel('Weekly Sales')
    plt.title(f'Weekly Sales Over Time for Store Type {type} with Predictions')
    
    return plt

# Streamlit app
st.set_page_config(page_title="Sales Prediction Web",page_icon=":chart_with_upwards_trend:",layout="wide")

# Header section
st.title("Walmart Sales Prediction:chart_with_upwards_trend:")
st.write("With this simple app, you can predict sales of Walmart by inputting the storetype and model")

# User inputs
#store = st.selectbox('Store', store_data['Store'].unique())
type = st.selectbox('Storetype', ['A', 'B', 'C'])
model = st.selectbox('Model', ['Linear Regression', 'Random Forest Regressor'])

# Make predictions and plot the results
if st.button('Make Prediction'):
    fig = make_prediction(type, model)
    st.pyplot(fig)