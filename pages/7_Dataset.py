import streamlit as st
import pandas as pd

@st.cache_data
def load_data():
    df = pd.read_excel(file)
    return df

file = "D:\Coding\python_projects\PolluCast\data\Dataset Clean.xlsx"
df = load_data()
total = len(df)

st.title("Dataset")
st.write("Total Data : ", total)
st.write(df)