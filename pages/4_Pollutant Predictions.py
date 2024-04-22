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

def load_models(station,polutan):
    model_folder = "models"  # Update with correct folder path
    model_name = f"st{station}_{polutan}.h5"
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

predictions = []

# List of pollutants to predict
pollutants = ["pm10", "so2", "co", "o3", "no2", "ispu"]

if st.sidebar.button("Predict", type='primary'):
    if (input_date != None and input_station != None):
        for pollutant in pollutants:
            # Load model
            model = load_models(input_station, pollutant)

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

        category = st.text

        if 0 < predictions[5] <= 50:
            category = "Baik"
        elif 50 < predictions[5] <= 100:
            category = "Sedang"
        elif 100 < predictions[5] <= 200:
            category = "Tidak Sehat"
        elif 200 < predictions[5] <= 300:
            category = "Sangat Tidak Sehat"
        elif 300 < predictions[5]:
            category = "Berbahaya"

        st.title("Pollutant Predictions")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("<h2 style='text-align: center'>{}</h2>".format(predictions[0]), unsafe_allow_html=True)
            st.markdown("<p style='text-align: center'>{}</p>".format(pollutants[0].upper()), unsafe_allow_html=True)

        with col2:
            st.markdown("<h2 style='text-align: center'>{}</h2>".format(predictions[1]), unsafe_allow_html=True)
            st.markdown("<p style='text-align: center'>{}</p>".format(pollutants[1].upper()), unsafe_allow_html=True)

        with col3:
            st.markdown("<h2 style='text-align: center'>{}</h2>".format(predictions[2]), unsafe_allow_html=True)
            st.markdown("<p style='text-align: center'>{}</p>".format(pollutants[2].upper()), unsafe_allow_html=True)

        with col1:
            st.markdown("<h2 style='text-align: center'>{}</h2>".format(predictions[3]), unsafe_allow_html=True)
            st.markdown("<p style='text-align: center'>{}</p>".format(pollutants[3].upper()), unsafe_allow_html=True)

        with col2:
            st.markdown("<h2 style='text-align: center'>{}</h2>".format(predictions[4]), unsafe_allow_html=True)
            st.markdown("<p style='text-align: center'>{}</p>".format(pollutants[4].upper()), unsafe_allow_html=True)

        with col3:
            st.markdown("<h2 style='text-align: center'>{}</h2>".format(predictions[5]), unsafe_allow_html=True)
            st.markdown("<p style='text-align: center'>{}</p>".format(pollutants[5].upper()), unsafe_allow_html=True)

        idx = predictions.index(max(predictions[0:5]))
        critical = pollutants[idx]

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("<p style='text-align: center; padding: 25px 0'>Critical</p>", unsafe_allow_html=True)
        with col2:
            st.markdown("<h2 style='text-align: center'>{}</h2>".format(critical.upper()), unsafe_allow_html=True)
        with col3:
            st.markdown("<p style='text-align: center; padding: 25px 0'>Category</p>", unsafe_allow_html=True)
        with col4:
            st.markdown("<h2 style='text-align: center'>{}</h2>".format(category), unsafe_allow_html=True)
    else:
        st.title("Pollutant Predictions")
        st.write("Please choose the date and station you want to predict")
else:
    st.title("Pollutant Predictions")