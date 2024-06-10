import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import pickle
import sklearn
from xgboost import XGBRegressor


st.set_page_config(layout="wide")
@st.cache_data(hash_funcs={XGBRegressor: id})
def load_model(filename):
    with open(filename, 'rb') as file:
        loaded_model = pd.read_pickle(file)
    return loaded_model


model = load_model('Model/finalized_model.pkl')


selected_value = st.sidebar.selectbox(
    "Menu",
    ("Home", "Price Predictor", "Model Performance")
)

if selected_value == "Home":

    st.title("Home LLC US Homes Price Predictor")
    col1, col2, col3 = st.columns(3)

    with col2:
        st.image('Images/home_llc.png',output_format='jpeg')

    st.write("**Home LLC has tasked me with developing a tool to predict home prices in the United States, taking into account various economic factors that influence the housing market. To create a comprehensive and accurate model, I consulted several authoritative sources, including data from the Federal Reserve of St. Louis, the World Bank, and the US Census Bureau.**")    

    st.title("Data used can be accessed from the below links")

    col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)

    with col1:
        st.link_button("Housing Prices", "https://fred.stlouisfed.org/series/CSUSHPISA")

    with col2:
        st.link_button("Federal Interest Rates", "https://fred.stlouisfed.org/series/FEDFUNDS")

    with col3:
        st.link_button("Economic Policy", "https://fred.stlouisfed.org/series/USEPUINDXD")

    with col4:
        st.link_button("Personal Income", "https://fred.stlouisfed.org/series/DSPIC96")

    with col5:
        st.link_button("Consumer Index", "https://fred.stlouisfed.org/series/CORESTICKM159SFRBATL")

    with col6:
        st.link_button("Employment Rate", "https://fred.stlouisfed.org/series/LNS14000024")

    with col7:
        st.link_button("Construction Materials", "https://fred.stlouisfed.org/series/WPUSI012011")

    with col8:
        st.link_button("Housing Completions", "https://fred.stlouisfed.org/series/COMPUTSA")    

    st.title("Sources Used for Research")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.link_button("The Top Factors Affecting House Prices", " https://www.theownteam.com/blog/the-top-factors-affecting-house-prices-and-how-to-navigate-them/")

    with col2:
        st.link_button("Factors Affecting the Real Estate Market", "https://www.investopedia.com/articles/mortages-real-estate/11/factors-affecting-real-estate-market.asp")

    with col3:
        st.link_button("Effects of economic factors on selling prices in U.S housing market", "https://www.sciencedirect.com/science/article/pii/S2666764923000383")

    with col4:
        st.link_button("Policy Uncertainty and House Prices in the United States", "https://www.researchgate.net/publication/317831218_Policy_Uncertainty_and_House_Prices_in_the_United_States")



if selected_value == "Price Predictor":

    st.title("**Put in the numbers and see the magic!**")

    col1,col2,col3 = st.columns([1,2,1])

    with st.form('Predict'):
        a = st.number_input('Interest Rate')
        b = st.number_input('Construction Materials')
        c = st.number_input('Income')
        d = st.number_input('Total Units')
        e = st.number_input('Inflation')
        f = st.number_input('Unemployment Rate')
        g = st.number_input('Year',format="%d", value=2004)
        h = st.number_input('Month', format="%d", value=1)

        submit = st.form_submit_button('Predict')

        if submit:
            input_data = np.array([[a, b, c, d, e, f, g, h]])
            prediction = model.predict(input_data)
            st.write(f"**The Case Schiller Index is {round(prediction[0],0)}**")

if selected_value == "Model Performance":

    st.title("Model performance in terms of R2 scores by different regression algorithms")

    st.image('Images/img.png',width=1000)
        


