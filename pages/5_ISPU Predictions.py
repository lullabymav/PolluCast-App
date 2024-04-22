import tensorflow as tf
import numpy as np
import streamlit as st
import pandas as pd
import datetime
import os

def load_data():
    df = pd.read_excel(file)
    return df

file = "data/Dataset Clean.xlsx"
df = load_data()

def load_models(station):
    model_folder = "models"  # Update with correct folder path
    model_name = f"st{station}_ispu.h5"
    model_path = os.path.join(model_folder, model_name)
    model = tf.keras.models.load_model(model_path)
    return model

input_date = st.sidebar.date_input(
    'Choose Date:',
    None,
    max_value=datetime.date(2022,1,1),
    key='date_input',
    format="YYYY-MM-DD"
)

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

predictions = []

# List of pollutants to predict
stations = [1,2,3,4,5]

if st.sidebar.button("Predict", type='primary'):
    if (input_date != None):
        for station in stations:
            # Load model
            model = load_models(station)

            # Load the dates
            df_st.index = df_st.index.astype(str)
            dates = np.array(df_st.index.values)

            # Extend the dates to include the prediction date
            dates_hat = dates.copy()
            current_date = pd.Timestamp(2022, 1, 2)
            end_date = pd.Timestamp(2022, 1, 3)

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
            predictions.append(np.round(prediction[0][0]))

        sum_pred = sum(predictions)
        avg_pred= sum_pred/len(predictions)

        category = st.text

        if 0 < avg_pred <= 50:
            category = "Baik"
        elif 50 < avg_pred <= 100:
            category = "Sedang"
        elif 100 < avg_pred <= 200:
            category = "Tidak Sehat"
        elif 200 < avg_pred <= 300:
            category = "Sangat Tidak Sehat"
        elif 300 < avg_pred:
            category = "Berbahaya"

        st.title("ISPU Predictions in DKI Jakarta")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("<h2 style='text-align: center'>{}</h2>".format(predictions[0]), unsafe_allow_html=True)
            st.markdown("<p style='text-align: center'>Jakarta Pusat</p>", unsafe_allow_html=True)

        with col2:
            st.markdown("<h2 style='text-align: center'>{}</h2>".format(predictions[1]), unsafe_allow_html=True)
            st.markdown("<p style='text-align: center'>Jakarta Utara</p>", unsafe_allow_html=True)

        with col3:
            st.markdown("<h2 style='text-align: center'>{}</h2>".format(predictions[2]), unsafe_allow_html=True)
            st.markdown("<p style='text-align: center'>Jakarta Selatan</p>", unsafe_allow_html=True)

        with col1:
            st.markdown("<h2 style='text-align: center'>{}</h2>".format(predictions[3]), unsafe_allow_html=True)
            st.markdown("<p style='text-align: center'>Jakarta Timur</p>", unsafe_allow_html=True)

        with col2:
            st.markdown("<h2 style='text-align: center'>{}</h2>".format(predictions[4]), unsafe_allow_html=True)
            st.markdown("<p style='text-align: center'>Jakarta Barat</p>", unsafe_allow_html=True)

        with col3:
            st.markdown("<h2 style='text-align: center'>{}</h2>".format(avg_pred), unsafe_allow_html=True)
            st.markdown("<p style='text-align: center'>DKI Jakarta</p>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<p style='text-align: right; padding: 25px 0; margin: 0 25px'>Category</p>", unsafe_allow_html=True)
        with col2:
            st.markdown("<h2 style='text-align: left'>{}</h2>".format(category), unsafe_allow_html=True)
    else:
        st.title("ISPU Predictions in DKI Jakarta")
        st.write("Please choose the date you want to predict")
else:
    st.title("ISPU Predictions in DKI Jakarta")