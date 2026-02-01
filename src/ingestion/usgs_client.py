import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

USGS_BASE_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"

def get_earthquakes(starttime, endtime, min_magnitude, bbox=None, limit=10):
    '''
    Get eathquakes events from USGS API through its query service 
    NOTE: more params can be added following the APIs documentation
    
    :param starttime: ISO DATE/TIME format
    :param endtime: ISO DATE/TIME format
    :param min_magnitude: Limit to events with a magnitude smaller than the specified maximum
    :param bbox: To select a specific region (tuple) -> {min_lon, min_lat, max_lon, max_lat}
    :param limit: how many event we want to get
    '''

    params = {
        'format': 'geojson', # always get this format
        'starttime': starttime,
        'endtime': endtime,
        'minmagnitude': min_magnitude,
        'limit':limit,
        'orderby': 'time',
    }

    if bbox != None:
        params.update(
            {'minlongitude': bbox[0],
             'minlatitude': bbox[1],
             'maxlongitude': bbox[2],
             'maxlatitude': bbox[3]}
            )
    
    response = requests.get(USGS_BASE_URL, params=params)
    response.raise_for_status()

    data = response.json()
    return parse_from_geojson(data)

def parse_from_geojson(data):
    '''
    Get the details from each earthquake from docstring
    
    :param data: dictionary with the information from usgs website
    '''
    
    records = []

    for feature in data['features']:
        #print('#########')
        #print(feature)
        properties = feature['properties']
        geometry = feature['geometry']

        # More interesting features can be obtained from each earthquake
        # Just added the initial ones

        record = {'time': datetime.utcfromtimestamp(properties['time']/1000),
                  'magnitude': properties['mag'],
                  'place': properties['place'],
                  'longitude': geometry['coordinates'][0],
                  'latitude': geometry['coordinates'][1],
                  'depth': geometry['coordinates'][2],
                  'event_id': feature['id']
                  }
        records.append(record)

    df = pd.DataFrame(records)
    df = df.sort_values('time').reset_index(drop=True)

    return df

def save_to_postgres(df, table_name, engine):
    '''
    Save the dataframe to a Postgres table

    :param df: DataFrame with the earthquake data
    :param table_name: Name of the table where to save the data
    :param engine: SQLAlchemy engine connected to the Postgres database
    '''
    df.to_sql(table_name,engine, if_exists='append', index=False)
    print(f"Data saved to table {table_name} successfully.")

def get_postgres_engine():
    '''
    Create a SQLAlchemy enginte to connect to Postgres with env vars
    '''
    username = os.getenv("POSTGRE_USERNAME")
    password = os.getenv("POSTGRE_PASSWORD")
    host = os.getenv("POSTGRE_HOST")
    port = os.getenv("POSTGRE_PORT")
    db = os.getenv("POSTGRE_DB")

    DATABASE_URL = f'postgresql://{username}:{password}@{host}:{port}/{db}'
    engine = create_engine(DATABASE_URL)
    return engine

def create_postgres_table(engine, table_name):
    '''
    Create the earthquakes table in Postgres if it does not exist
    '''
    create_table_query = f'''
    CREATE TABLE IF NOT EXISTS {table_name} (
        event_id TEXT PRIMARY KEY,
        time TIMESTAMP,
        magnitude FLOAT,
        place TEXT,
        longitude FLOAT,
        latitude FLOAT,
        depth FLOAT
    );
    '''
    with engine.connect() as connection:
        connection.execute(text(create_table_query))
    print(f"Table {table_name} is ready.")

def main():
    end = datetime.now(timezone.utc) # type: ignore
    start = end -  timedelta(days=30)
    starttime = start.strftime('%Y-%m-%d')
    endtime = end.strftime('%Y-%m-%d')
    table_name = 'earthquakes'

    data = get_earthquakes(starttime=starttime, endtime=endtime, min_magnitude=6,limit= 100)
    print_df_info(data)

    engine = get_postgres_engine()
    create_postgres_table(engine, table_name)
    save_to_postgres(data, table_name, engine)
    print_sql_info(table_name, engine)
    print("Data extraction completed successfully.")


def print_df_info(df):
    print("DataFrame Info:")
    print(df.info())
    print("\nDataFrame Head:")
    print(df.head())

def print_sql_info(table_name, engine):
    query = f"SELECT * FROM {table_name} LIMIT 5;"
    with engine.connect() as connection:
        result = connection.execute(text(query))
        for row in result:
            print(row)

if __name__ == "__main__":
    main()
