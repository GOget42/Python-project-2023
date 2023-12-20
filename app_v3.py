import streamlit as st
from joblib import load
import pandas as pd

# Load the model from the file
rf = load('rf_model.joblib')

# Load the store data
store_data = pd.read_csv('stores.csv')

st.set_page_config(page_title="Sales Prediction Web",page_icon=":chart_with_upwards_trend:",layout="wide")

#Header section
st.title("Walmart Sales Prediction:chart_with_upwards_trend:")
st.write("With this simple app, you can predict sales of Walmart by inputting the store, type, and size")

# User inputs
store = st.selectbox('Store', store_data['Store'].unique())
type = st.selectbox('Type', ['A', 'B', 'C'])
size = st.number_input('Size', min_value=1)

# Get the store data
store_row = store_data[(store_data['Store'] == store) & (store_data['Type'] == type)]

# Check if the store exists
if len(store_row) == 0:
    st.write('The selected store does not exist.')
else:
    # Check the size
    if size != store_row['Size'].values[0]:
        st.write('The input size does not match the size of the selected store.')
    else:
        # Create a new DataFrame with the input features
        new_data = pd.DataFrame({
            'Store': [store],
            'Type': [type],
            'Size': [size],
            # include any other features used in the model
        })

        # Use the model to make a prediction
        new_prediction = rf.predict(new_data)

        # Display the prediction
        st.subheader(f"The estimated sales of Walmart is :{new_prediction[0]:.2f}")


