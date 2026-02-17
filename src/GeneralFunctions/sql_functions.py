
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os

load_dotenv()

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

def save_to_postgres(df, table_name, engine):

    cols = ', '.join(df.columns)
    placeholders = ', '.join(f':{col}' for col in df.columns)
    sql = text(f'INSERT INTO {table_name} ({cols}) VALUES ({placeholders}) ON CONFLICT (event_id) DO NOTHING')
    with engine.begin() as conn:
        conn.execute(sql, df.to_dict(orient='records'))
    print(f"Data saved to table {table_name} successfully.")


def create_postgres_table(engine, table_name, query):
    '''
    Create the earthquakes table in Postgres
    we need this function to avoid repeating values and solve conflicts
    this table's structure will be the same everytime, that is why 
    we can hardcode it here
    '''
    with engine.begin() as connection:
        connection.execute(text(query))
    print(f"Table {table_name} is ready.")


def print_sql_info(table_name, engine, limit=5):
    query = f"SELECT * FROM {table_name} LIMIT {limit};"
    with engine.connect() as connection:
        result = connection.execute(text(query))
        for row in result:
            print(row)


def read_from_sql(table_name: str, engine, limit=None) -> pd.DataFrame:
    if limit is not None:
        query = f'SELECT * FROM {table_name} LIMIT {limit};'
    else:
        query = f'SELECT * FROM {table_name};'
    with engine.connect() as conn:
        df = pd.read_sql_query(query, conn)
    return df