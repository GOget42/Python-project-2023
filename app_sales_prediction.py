#import the libraries and streamlit to generate simple interface
import streamlit as st
#import ML model here
#import numpy as np
#import pandas as pd

st.set_page_config(page_title="Sales Prediction Web",page_icon=":chart_with_upwards_trend:",layout="wide")

#Header section
st.title("Walmart Sales Prediction:chart_with_upwards_trend:")
st.write("With this simple app, you can predict sales of Walmart by inputting the time and other variables")
#features: what variables to enter to extract info from model;other input can be added based on the ML model
input_time = st.slider("Input the time(in year)",2024,2034)

click= st.button("Predict sales")
if click:
        #strat the prediction(wirrten in ML model)
        #example:
        #predictsales = 1000
#the predictsales will be the sales calculated by ML model
st.subheader(f"The estimated sales of Walmart is :{predictsales:.2f}")