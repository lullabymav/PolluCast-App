import datetime
import streamlit as st
import pandas as pd

@st.cache_data
def load_data():
    df = pd.read_excel(file)
    return df

file = "D:\Coding\python_projects\PolluCast\data\Dataset Clean.xlsx"
df = load_data()

def graph_per_station(number,start,end) :
  st_filter = df.loc[(df['stasiun'] == number) & (df['tanggal'] >= start) & (df['tanggal'] <= end)]
  cols = st_filter.columns
  par = cols[2:]
  for col in par:
      st.write("### {0} on station {1}".format(col, number))
      st.line_chart(
          data=st_filter,
          x='tanggal',
          y=col,
          color="#ffaa00",
          width=1000,
          height=500,
          use_container_width=True
      )

# Sidebar text input for station number
station_number = st.sidebar.selectbox(
    'Station:', ("Jakarta Pusat","Jakarta Utara","Jakarta Selatan","Jakarta Timur","Jakarta Barat"),
    index=None,
    placeholder='choose station'
)

# Sidebar text input for start date
start_date = st.sidebar.date_input(
    'Start Date:',
    None,
    min_value=datetime.date(2019,1,1),
    max_value=datetime.date(2021,12,31),
    key='start',
    format="YYYY-MM-DD"
)

# Sidebar text input for end date
end_date = st.sidebar.date_input(
    'End Date:',
    None,
    min_value=datetime.date(2019,1,1),
    max_value=datetime.date(2021,12,31),
    key='end',
    format="YYYY-MM-DD"
)

if station_number==None or start_date==None or end_date==None:
    st.title("Pollutant and Meteorological Graphs 2019 - 2021")
else:
    # Convert station number to integer
    if(station_number=="Jakarta Pusat"):
        station_number = 1
    elif(station_number=="Jakarta Utara"):
        station_number = 2
    elif(station_number=="Jakarta Selatan"):
        station_number = 3
    elif(station_number=="Jakarta Timur"):
        station_number = 4
    elif(station_number=="Jakarta Barat"):
        station_number = 5
    start_date = start_date.strftime("%Y-%m-%d")
    end_date = end_date.strftime("%Y-%m-%d")
    graph_per_station(station_number, start_date, end_date)
