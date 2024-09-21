
import pathlib
import datetime
import pandas as pd


SHIFT_START_TIME = datetime.time(hour=7, minute=45)


def calculate_machine_availability(df: pd.DataFrame):
    total_slot_length = 3600 * 8.75  # 8.75 hours in seconds
    df = (df.groupby(['CNC', 'date'])['t_total'].sum().dt.total_seconds() / total_slot_length) * 100
    return df.reset_index(name='availability').rename(columns={'t_total': 'availability'})


def calculate_time_between_programs(df: pd.DataFrame):
    df = df.assign(
        group=(df['dimension'] != df['dimension'].shift()).cumsum()
    )
    groups = df.groupby(['CNC', 'group', 'dimension']).agg({'dt_stop': 'max', 'dt_start': 'min'}).reset_index()
    time_between = (groups['dt_start'].shift(-1) - groups['dt_stop']).dt.total_seconds() / 60
    return groups.assign(time_between=time_between)


def calculate_warmup_time(df: pd.DataFrame, shift_start: datetime.time=None):
    """
    Here they want to measure time lost from start of shift to the start of the first run.
    How it should be calculated is Start time column of the first programme minus 07:45.
    The y-axis should be labelled in minutes. There should be one result per day similar to availability.
    """

    hour, minute = (shift_start.hour, shift_start.minute) if shift_start else (SHIFT_START_TIME.hour, SHIFT_START_TIME.minute)
    
    def _compute(x: pd.DataFrame):
        start: pd.Timestamp = min(x)
        diff =  start - pd.Timestamp(hour=hour, minute=minute, year=start.year, month=start.month, day=start.day)
        return diff.total_seconds() / 60

    warmup_df = df.groupby(['CNC', 'date'])['dt_start'].apply(_compute)
    return warmup_df.reset_index(name='warmup_time')


def first_part_start_and_end_time(df: pd.DataFrame):
    return df.groupby(['CNC', 'date']).agg(
        {'dt_start': 'min', 'dt_stop': 'max', 'dimension': len}
    ).reset_index()


def summary_statistics(df: pd.DataFrame):
    dfx = first_part_start_and_end_time(df).rename(
        columns={
            'dt_start': 'Start Time',
            'dt_stop': 'Stop Time',
            'dimension': 'Total Quantity Produced'
        }
    )
    dfx['Data Sample Size'] = len(df)

    summary = dict(
        summary = dfx,
        most_produced_program = df[['CNC', 'dimension', 'prgm']].value_counts().head(),
    )
    return summary


def calculate_productivity(df: pd.DataFrame):
    """
    productivity is the sum(t_average) / sum(quantity) for each dimension, date
    """
    prod = df.groupby(['CNC', 'dimension', 'date']).apply(
        lambda d: (d['t_average'].dt.total_seconds() / 60).sum() / d['quantity'].sum()
    ).reset_index(name='productivity')
    return prod


def to_timedelta(df, prefix: str):
    return pd.Timedelta(
        hours=df[prefix + 'Hour'],
        minutes=df[prefix + 'Min'],
        seconds=df[prefix + 'Sec']
    )


def concatenate_dimensions(df: pd.DataFrame):
    df['dimension'] = df['dimX'].astype(str) + 'x' + df['dimY'].astype(str) + 'x' + df['dimZ'].astype(str)
    return df.drop(columns=['dimX', 'dimY', 'dimZ'])


def convert_to_timedelta(df, col_prefix: str):
    df['t_' + col_prefix] = df.apply(to_timedelta, prefix=col_prefix, axis=1)
    # drop the cols
    return df.drop([col_prefix + i for i in ('Hour', 'Min', 'Sec')], axis=1)


def convert_to_timestamp(df, col_prefix: str):
    df['dt_' + col_prefix] =\
        pd.to_datetime(
           df['date'] + ' ' + df[col_prefix + 'Hour'].astype(str) + ':' + df[col_prefix + 'Min'].astype(str) + ':' + df[col_prefix + 'Sec'].astype(str)
        )
    return df.drop([col_prefix + i for i in ('Hour', 'Min', 'Sec')], axis=1)


def load_csv(filename):
    header = [
        'area',
        'prgm',
        'description',
        'dimX',
        'dimY',
        'dimZ',

        'startHour',
        'startMin',
        'startSec',

        'stopHour',
        'stopMin',
        'stopSec',

        'effectiveHour',
        'effectiveMin',
        'effectiveSec',

        'totalHour',
        'totalMin',
        'totalSec',

        'quantity',
        'averageHour',
        'averageMin',
        'averageSec'
    ]

    date = pathlib.Path(filename).name.split('.').pop(0)
    df = pd.read_csv(filename, names=header, skiprows=1, index_col=False)
    df['CNC'] = 'CNC1'
    df['date'] = date
    df = concatenate_dimensions(df)
    return df


def read_vpro_data(file):
    return clean_data(load_csv(file))


def clean_data(df: pd.DataFrame):
    
    df['prgm'] = df['prgm'].str.split('\\').apply(lambda i: i[-1])

    # filter rows with quantity zero
    df = df[df['quantity'] > 0]

    # filter rows with invalid date, redudant
    df = df[
        (df['startHour'] <= 23) &
        (df['startMin'] <= 60) &
        (df['startSec'] <= 60)
    ]

    df = df[
        (df['stopHour'] <= 23) &
        (df['stopMin'] <= 60) &
        (df['stopSec'] <= 60)
    ]

    # convert (xHour, xMin, xSec) to one timedelta object
    for col in ['average', 'total', 'effective']:
        df = convert_to_timedelta(df, col)

    # convert (xHour, xMin, xSec) to one datetime object
    for col in ['start', 'stop']:
        df = convert_to_timestamp(df, col_prefix=col)

    df = df.assign(date=pd.to_datetime(df['date'])).sort_values(by='dt_start')

    return df


df = read_vpro_data('datasource/CNC_1/2024/20240118.pro')