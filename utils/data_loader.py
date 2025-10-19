import pandas as pd

def load_airports(path="data/raw/airports.dat"):
    df = pd.read_csv(path, header=None, names=[
        "id","name","city","country","iata","icao","lat","lon","alt","tz","dst","tzdb"
    ])
    df = df[df['iata'].notnull()]
    return df

def load_routes(path="data/raw/routes.dat"):
    df = pd.read_csv(path, header=None, names=[
        "airline","airline_id","src","src_id","dst","dst_id","codeshare","stops","equipment"
    ])
    return df
