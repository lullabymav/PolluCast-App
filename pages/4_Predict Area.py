import tensorflow as tf
import numpy as np
import streamlit as st
import pandas as pd
import datetime
import os

def load_data():
    df = pd.read_excel(file)
    return df

file = "D:\Coding\python_projects\PolluCast\data\Dataset Clean.xlsx"
df = load_data()

def load_models(station,polutan):
    model_folder = "D:\Coding\python_projects\PolluCast\models"  # Update with correct folder path
    model_name = f"st{station}_{polutan}.h5"
    model_path = os.path.join(model_folder, model_name)
    return model_path

input_date = st.sidebar.date_input(
    'Choose Date:',
    None,
    max_value=datetime.date(2022,1,1),
    key='date_input',
    format="YYYY-MM-DD"
)
input_station = st.sidebar.selectbox(
    'Station:', ("Jakarta Pusat","Jakarta Utara","Jakarta Selatan","Jakarta Timur","Jakarta Barat"),
    index=None,
    placeholder='choose station'
)

if (input_station == "Jakarta Pusat"):
    input_station = 1
elif (input_station == "Jakarta Utara"):
    input_station = 2
elif (input_station == "Jakarta Selatan"):
    input_station = 3
elif (input_station == "Jakarta Timur"):
    input_station = 4
elif (input_station == "Jakarta Barat"):
    input_station = 5

def df_to_X_y(df, window_size=1):
    df_as_np = df.to_numpy()
    X = []
    y = []
    for i in range(len(df_as_np)-window_size):
        row = [r for r in df_as_np[i:i+window_size]]
        X.append(row)
        label = df_as_np[i+window_size][0]
        y.append(label)
    return np.array(X), np.array(y)

df_st = df.loc[df['stasiun'] == 1].reset_index()
df_st = df_st.drop(['index','stasiun'], axis=1).set_index(['tanggal'])
X, y = df_to_X_y(df_st)

prediction = []

if(input_date != None and input_station != None):
    # List of pollutants to predict
    pollutants = ["pm10", "so2", "co", "o3", "no2", "ispu"]

    for pollutant in pollutants:
        # Load model
        model = load_models(input_station,pollutant)
        model = tf.keras.models.load_model(model)

        # Load the dates
        df_st.index = df_st.index.astype(str)
        dates = np.array(df_st.index.values)

        # Extend the dates to include the prediction date
        dates_hat = dates.copy()
        current_date = datetime.date(2022, 1, 2)
        end_date = datetime.date(2022, 1, 3)

        while current_date < end_date:
            dates_hat = np.append(dates_hat, current_date.strftime('%Y-%m-%d'))
            current_date += pd.DateOffset(days=1)

        # Convert dates to indices
        date_indices = {tuple(date): i for i, date in enumerate(dates_hat)}

        # Convert input date to index
        input_index = date_indices[tuple(input_date.strftime("%Y-%m-%d"))]

        # Prepare input data for prediction
        input_data = np.array(X[input_index - 1])
        input_data = input_data.reshape(-1, input_data.shape[0], input_data.shape[1], 1)

        prediction = model.predict(input_data)
        prediction.append(prediction)
else:
    st.title("Pollutant Forecast")