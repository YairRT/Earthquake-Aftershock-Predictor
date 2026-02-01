import pandas as pd 
import numpy as np

def read_from_sql(table_name: str, engine) -> pd.DataFrame:
    query = f'SELECT * FROM {table_name};'
    with engine.connect() as conn:
        df = pd.read_sql_query(query, conn)
    return df

def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Basic preprocessing steps
    - Convert data types
    - Handle missing values
    '''
    out = df.copy()

    # Convert data types
    out['time'] = pd.to_datetime(out['time'], utc=True, errors='coerce')
    out['magnitude'] = pd.to_numeric(out['magnitude'], errors='coerce')
    out['depth'] = pd.to_numeric(out['depth'], errors='coerce')
    out['latitude'] = pd.to_numeric(out['latitude'], errors='coerce')
    out['longitude'] = pd.to_numeric(out['longitude'], errors='coerce')

    # Handle missing values - for simplicity, drop rows with missing critical values
    out = out.dropna(subset=['event_id', 'time', 'magnitude', 'latitude', 'longitude', 'depth'])

    return out

def basic_time_feats(df: pd.DataFrame):
    '''
    Docstring for basic_time_feats

    :type df: dataframe type with time columns (dayOfWeek, hour, etc)
    '''

    df['hour'] = df['time'].dt.hour # type: ignore
    df['dayofweek'] = df['time'].dt.dayofweek # type: ignore # Monday = 0
    return df

def compute_freq_series(df: pd.DataFrame, freq: str = 'D'):
    '''
    Docstring for compute_freq_series
    
    :param df: earthquake df
    :param freq: 'Hourly', 'D' daily
    :returns: total counts pd.Dataframe
    '''

    if df.empty:
        return pd.DataFrame({'time':[],'count':[]})
    

    counts = df.set_index('time').resample(freq).size().reset_index(name='count')

    return counts

def add_seq_feat(df: pd.DataFrame):
    '''
    Docstring for add_seq_feat
    
    Adds simple per-event 'sequence' features that we'll use for logistic reg.
        - time since_prev_hours
        - distance_to_prev_km
        - rolling_count_6hr, rolling_count_24h (based on event times)
    '''
    if df.empty:
        return df

    # Time since previous event (hours)
    dt = df['time'].diff().dt.total_seconds() / 3600.0 # type: ignore
    df['time_since_prev_hours'] = dt.fillna(np.nan)

    # distance to previous event (km) using haversine
    df['distance_to_prev_km'] = haversine_km_convert(
        df['latitude'].shift(1),
        df['longitude'].shift(1),
        df['latitude'],
        df['longitude']
    )

    # Rolling counts how many quakes happened in the last X hours before each event
    # A rolling window aligned to each event timestam
    df = df.sort_values('time').set_index('time')
    df['rolling_count_6h'] = df['event_id'].rolling('6h').count().shift(1)
    df['rolling_count_24h'] = df['event_id'].rolling('24h').count().shift(1)
    df = df.reset_index()

    df['rolling_count_6h'] = df['rolling_count_6h'].fillna(0).astype(int)
    df['rolling_count_24h'] = df['rolling_count_24h'].fillna(0).astype(int)

    return df

def depth_stats(df: pd.DataFrame):
    '''
    Write some statistics regarding depth
    '''
    if df.empty:
        return {'mean': None, 'p50': None, 'p90': None, 'max': None}
    
    return {'mean': float(df['depth'].mean()), 'p50': float(df['depth'].quantile(0.50)), 'p90': float(df['depth'].quantile(0.90)), 'max': float(df['depth'].max())}

def mag_stats(df: pd.DataFrame):
    '''
    Write some statis regarding mag
    '''
    if df.empty:
        return {'mean': None, 'p50': None, 'p90': None, 'max': None}
    
    return {'mean': float(df['magnitude'].mean()), 'p50': float(df['magnitude'].quantile(0.50)), 'p90': float(df['magnitude'].quantile(0.90)), 'max': float(df['magnitude'].max())}


# Now let's create a function to calculate distance in Km in between two geospatial points on earth
# Haversine is a good option for this purpose

def haversine_km_convert(lat1, lon1, lat2, lon2):
    '''
    Vector of Haversine distances transformed into KM
    '''

    R = 6371.0 # Radius of Earth in KMm

    # Convert from degrees to rad
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # This formula can also be used but for numerical stability is better the other one
    # a = (1 - np.cos(dlat) + np.cos(lat1)* np.cos(lat2)*(1-np.cos(dlon)))/2 
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a)) # to calculate for the great-circle distance
    return R * c


def main():
    from datetime import datetime, timedelta, timezone
    from src.ingestion.usgs_client import get_postgres_engine

    table_name = 'earthquakes'

    days=30

    print('Reading data from Postgres...')

    engine = get_postgres_engine()
    df = read_from_sql(table_name, engine)
    print(f"Total records retrieved from SQL: {len(df)}")

    print('Data from Postgres read successfully. Preprocessing...')

    df = preprocess_df(df)
    print(f"Total records after preprocessing: {len(df)}")

    print('Preprocessing done. Generating features...')


    print('Basic time features:')
    seq = basic_time_feats(df)
    print(seq.head())

    print('\nFrequency series (daily)')
    print(compute_freq_series(df).head())

    print('\nSequence features:')
    seq = add_seq_feat(seq)
    print(seq.head())
    print(seq.columns)

    print('\nStatistics of depth')
    print(depth_stats(df))

    print('\nStatistics of magnitude')
    print(mag_stats(df))

    print('Feature generation completed successfully.')

if __name__=='__main__':
    main()


# Next thing to do:
# - Create a notebook to test these features (check what is this)
# - Create unit tests for these functions (check what is this)
# - Save thiese features into a new table in Postgres