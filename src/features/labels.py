import pandas as pd
import numpy as np

from src.features.features import haversine_km_convert

def add_aftershock_label(df: pd.DataFrame, T_hours: float = 24.0, R_km: float = 100.0,
                         min_aftershock_mag: float | None = None):
    '''
    Adds binary label 'y_aftershock':
    y=1 if there exists a future event within the next T_hours AND R_km km
    '''
    out = df.copy()

    if out.empty:
        out['y_aftershock'] = []
        return out
    
    # At this point, the NaNs of Lat, Lon and time should already be dropped 
    # The df should already be sorted time-wise

    times = out['time'].to_numpy()
    lats = out['latitude'].astype(float).to_numpy()
    lons = out['longitude'].astype(float).to_numpy()
    mags = out['magnitude'].to_numpy()

    y = np.zeros(len(out),dtype=int)

    # Let's create a window T_hours into the future for events within R_km
    for i in range(len(out)):
        # find end index for which time diff > T_hours
        k = i + 1
        while k < len(out) and (times[k] - times[i]) / np.timedelta64(1, 'h') <= T_hours:
            k+=1
        
        if k == i + 1:
            continue # There are no event within the desired time window

        # Compute distances from event i to i+1,...,k
        cand_idx = slice(i+1,k)
        
        lat_i = pd.Series([lats[i]] * (k - (i + 1)))
        lon_i = pd.Series([lons[i]] * (k - (i + 1)))
        lat_j = pd.Series(lats[cand_idx])
        lon_j = pd.Series(lons[cand_idx])

        dists = haversine_km_convert(lat_i, lon_i, lat_j, lon_j).to_numpy()

        if min_aftershock_mag is not None:
            cand_mags = mags[cand_idx]
            acceptable_mag = np.nan_to_num(cand_mags, nan=-999.0) >= float(min_aftershock_mag)
            hit = np.any((dists<=R_km) & acceptable_mag)
        else:
            hit = np.any(dists<=R_km)
        y[i] = 1 if hit else 0
    out['y_aftershock'] = y
    return out


if __name__=='__main__':
    # quick test
    from src.GeneralFunctions.sql_functions import get_postgres_engine, read_from_sql, create_postgres_table, save_to_postgres

    table_name_label = 'earthquake_labels'
    
    query = f'''
    CREATE TABLE IF NOT EXISTS {table_name_label} (
        event_id TEXT PRIMARY KEY,
        time TIMESTAMP,
        magnitude FLOAT,
        place TEXT,
        longitude FLOAT,
        latitude FLOAT,
        depth FLOAT,
        hour INT,
        dayofweek INT,
        time_since_prev_hours FLOAT,
        distance_to_prev_km FLOAT,
        rolling_count_6h INT,
        rolling_count_24h INT,
        y_aftershock INT
    );'''


    engine = get_postgres_engine()
    df = read_from_sql('earthquake_features', engine, limit=1000)
    out = add_aftershock_label(df, T_hours=24.0, R_km=100.0, min_aftershock_mag=4.0)

    create_postgres_table(engine, table_name_label, query)
    save_to_postgres(out, table_name_label, engine)



    print(out.head(20)[['place','y_aftershock']])




