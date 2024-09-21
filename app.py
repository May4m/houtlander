import warnings
warnings.filterwarnings("ignore")

import os
import time
import pathlib
import datetime
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import hashlib

import etl

import logging
import sys


logging.basicConfig(
    level=logging.INFO,  # Set the minimum log level
    format='%(asctime)s - %(levelname)s - %(message)s',  # Define the log message format
    handlers=[
        logging.StreamHandler(sys.stdout)  # Set the output stream to stdout
    ]
)


logger = logging.getLogger(__file__)


# Get the value of a specific config setting
development_mode = bool(os.environ.get('DEV_MODE', False))

DIV_COLOR = 'orange'
CACHE_TTL = 60 * 3  # two minte
CACHE_MAX_ENTRIES = 2  # cache max of 2 item to save memory


@st.cache_data(ttl=CACHE_TTL + 60 * 2)
def load_csv_file(file: str):
    """
    caches opened csv files, so when different files are selected they are not
    reloaded again
    """
    t0 = time.time()
    df = etl.load_csv(filename=file)
    duration = time.time() - t0
    logger.info("took {:.2f} seconds to open {}".format(duration, file))
    return df

# Function to fetch data from FTP (dummy implementation)
@st.cache_data(ttl=CACHE_TTL, max_entries=CACHE_MAX_ENTRIES)  # Cache for 10 minutes
def fetch_data_from_source(
    cnc_machine='CNC_1',
    start: datetime.datetime=None,
    end: datetime.datetime=None
):
    # TODO: add CNC parameter for selecting cnc

    base_path = f'datasource/{cnc_machine}/{start.year}'
    if (root_dir := os.environ.get('DATA_PATH')):
        base_path = f"{root_dir}/{base_path}"
    logger.info(f"Reading data from {base_path}")
    if start or end:
        files = list(pathlib.Path(base_path).glob('*.pro'))
        df = pd.DataFrame(
            {
                'files': files,
                'date': pd.to_datetime([i.name.split('.')[0] for i in files])
            }
        )

        # filter the dates before reading the files
        if start:
            df = df[df['date'] >= pd.to_datetime(start)]
        if end:
            df = df[df['date'] <= pd.to_datetime(end)]
        logger.info(f"reading a total of: {len(df)} files")

        # load the files
        data = [load_csv_file(i) for i in df['files']]

        # process the data
        full_df = etl.clean_data(pd.concat(data))
        full_df['CNC'] = cnc_machine

        return full_df

    return etl.read_vpro_data('2024/20240118.pro')


# New function to create line chart and histogram
def create_line_and_histogram(
        df: pd.DataFrame, x, y, color, title, x_label, y_label, bar=False, hist_color=None
):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=False, 
                        vertical_spacing=0.25, 
                        subplot_titles=(f"{title} - Time Series", f"{title} - Distribution"))
    
    # Line chart
    for cnc in df[color].unique():
        df_cnc = df[df[color] == cnc]
        if bar:
            fig.add_trace(
                go.Bar(x=df_cnc[x], y=df_cnc[y], name=cnc),
                row=1, col=1
            )
        else:
            fig.add_trace(
                go.Scatter(x=df_cnc[x], y=df_cnc[y], name=cnc, mode='lines+markers'),
                row=1, col=1
            )
    # Histogram
    num_bins = st.slider(
        'Select number of bins for histogram:', min_value=1, max_value=50, value=10, key=title,
        help="Adjust the number of bins for the histogram, i.e resolution of the histograms"
    )
    hist_color = hist_color if hist_color else color
    for cnc in df[hist_color].unique():
        df_cnc = df[df[hist_color] == cnc]
        fig.add_trace(go.Histogram(x=df_cnc[y], name=cnc, nbinsx=num_bins), row=2, col=1)
    
    fig.update_layout(height=800, title_text=title, xaxis_title=x_label, yaxis_title=y_label)
    fig.for_each_annotation(lambda a: a.update(font=dict(size=20, color='blue')))
    return fig


# Authentication functions and user database (unchanged)
def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()


def check_hashes(password, hashed_text):
    if make_hashes(password) == hashed_text:
        return hashed_text
    return False


users = {"houtlander": make_hashes("2020@houtlander")}


# add user account for dev mode for debugging
if development_mode:
    users["admin"] = make_hashes("admin")


# Login and logout functions (unchanged)
def login():
    st.sidebar.title("Login")
    username = st.sidebar.text_input("Username", help="default username is 'admin', default password is 'admin'")
    password = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        if username in users and check_hashes(password, users[username]):
            st.session_state.logged_in = True
            st.session_state.username = username
            st.sidebar.success(f"Logged in as {username}", icon="✅")
        else:
            st.sidebar.error("Incorrect username or password", icon="🚨")

def logout():
    st.session_state.logged_in = False
    st.session_state.username = None
    st.sidebar.success("Logged out successfully")


@st.dialog("🚨 error")
def modal(msg: str):
    st.write(msg)


# Main app function
def main():
    logger.info(f"Running in app development mode: {development_mode}")

    st.set_page_config(page_title="Daily Report", layout="wide")
    st.sidebar.image(
        'https://houtlander.co.za/cdn/shop/files/Final_Houtlander_Logo-01_32ae13f4-56e0-4631-84d4-25e8d9a65dc3_165x.jpg',
        width=180
    )  # Adjust the width as needed
    st.markdown(
        r"""
        <style>
        .stAppDeployButton {
                visibility: hidden;
            }
        </style>
        """, unsafe_allow_html=True
    )
    
    # Initialize session state for login
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.username = None

    # Show login/logout in sidebar
    if not st.session_state.logged_in:
        login()
    else:
        st.sidebar.button("Logout", on_click=logout)

    # Main app content
    if st.session_state.logged_in:
        st.title(f"Houtlander Operation Report - Welcome, {st.session_state.username}!")

        # Add refresh button
        if st.button("Refresh Data", help="Refresh when new data is available in google drive"):
            st.cache_data.clear()
            st.success("Data refreshed successfully!")


        # CNC type selection
        cnc_types = ['CNC_1', 'CNC_2']
        selected_cnc = st.sidebar.selectbox("Select CNC Type", cnc_types)
        
        # Sidebar for date range and CNC selection
        st.sidebar.header("Filters")
        start_date = st.sidebar.date_input(
            "Start Date", pd.Timestamp.now(),
            help="select the start date for the report. This is used to filter older data"
        )
        end_date = st.sidebar.date_input(
            "End Date", pd.Timestamp.now(),
            help="select the start date for the report"
        )
        if start_date > end_date:
            return modal("Please make sure Start Date is not less than End Date")

        try:
            data = fetch_data_from_source(selected_cnc, start_date, end_date)
        except ValueError:
            return modal(
                "No Data Available for the selected dates, please SELECT dates with data "
                "On the panel located at the right of the page"
            )
        
        
        if len(data) == 0:
            return modal("No Data Available for the select dates")


        # Filter data based on selected date range and CNC type
        mask = (data['date'] >= pd.Timestamp(start_date)) & (data['date'] <= pd.Timestamp(end_date))
        
        # TODO: make selected_cnc read correct data
        #if selected_cnc != 'All':
        #    mask &= (data['CNC'] == selected_cnc)
        filtered_data = data.loc[mask]

        # 1. Daily trend of machine availability per CNC
        st.header("Daily Trend of Machine Availability per CNC", divider=DIV_COLOR)
        # availability_data = calculate_machine_availability(filtered_data)
        availability_data = etl.calculate_machine_availability(filtered_data)
        if len(availability_data) == 1:
            data = availability_data.copy()
            data['availability'] = 100 - data['availability']
            data['date'] = 'Other'
            fig = px.pie(
                pd.concat([data, availability_data]),
                values='availability', names='date',
                title="Availability for the day is {:.1f}%".format(availability_data.availability[0]),
                color_discrete_sequence=['red', 'white']
            )
            st.plotly_chart(fig, theme=None)
        else:
            fig_availability = create_line_and_histogram(
                availability_data, 'date', 'availability', 'CNC', "Machine Availability %",
                x_label='Date', y_label='Availability(%)'
            )
            st.plotly_chart(fig_availability, use_container_width=True)

        # 2. Time between programs
        st.header("Time Between Programs", divider=DIV_COLOR)
        time_between_data = etl.calculate_time_between_programs(filtered_data)
        fig_time_between = create_line_and_histogram(
            time_between_data, 'dimension', 'time_between', 'CNC', "Time Between Programs (minutes)",
            x_label='Dimension ID', y_label='Time(Minutes)', bar=True
        )
        st.plotly_chart(fig_time_between, use_container_width=True)


        # 2. productivity = t_average
        st.header("Machine Productivity", divider=DIV_COLOR)
        productivity_data = etl.calculate_productivity(filtered_data)
        fig_time_between = create_line_and_histogram(
            productivity_data, 'date', 'productivity', 'dimension', "Machine Productivity",
            x_label='Productivity (Average Time/Quantity)', y_label='Quantity/Min', hist_color='CNC'
        )
        st.plotly_chart(fig_time_between, use_container_width=True)


        # 3. calculate_warmup_time
        st.header("Warmup Time Program", divider=DIV_COLOR)
        shift_start = st.sidebar.time_input(
            "Shift Start Time", datetime.time(7, 45),
            help="Shift Start time for the current report, this is used to calculate the warmup graph. Warmup is"
            " defined as `time when first item is produced` - `shift start time`"
        )
        warmup_time = etl.calculate_warmup_time(filtered_data, shift_start=shift_start)
        fig_time_between = create_line_and_histogram(
            warmup_time, 'date', 'warmup_time', 'CNC', "Warmup Time (minutes)",
            x_label='Date', y_label='Warmup Time(Minutes)'
        )
        st.plotly_chart(fig_time_between, use_container_width=True)

        # 4 & 5. Start time of first part and End time of last part
        st.header("Basic Summary Info", divider=DIV_COLOR)
        summary = etl.summary_statistics(filtered_data)
        st.subheader("Summary")
        st.dataframe(summary['summary'])
        
        st.subheader("Most Produced Products")
        st.dataframe(summary['most_produced_program'])

        st.subheader("Time Between Programs")

        st.dataframe(
            time_between_data.assign(
                date=time_between_data['dt_stop'].dt.date
            ).groupby(['date'])['time_between'].describe().T
        )
    else:
        st.title("Houtlander Operation Report")
        st.write("Please log in to view the report.")

if __name__ == "__main__":
    main()
