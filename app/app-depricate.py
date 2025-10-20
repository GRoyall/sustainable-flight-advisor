# ===========================
# Sustainable Flight Advisor MVP
# ===========================

# --- 1) Imports ---
import pandas as pd
import math
import numpy as np
import streamlit as st

# ===========================
# 2) Load Airports & Routes
# ===========================

# --- Airports ---
airports_url = "https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat"

colnames = [
    "id","name","city","country","iata","icao",
    "lat","lon","alt","tz","dst","tzdb","type","source"
]

airports = pd.read_csv(
    airports_url,
    header=None,
    names=colnames,
    quotechar='"',
    skipinitialspace=True
)

airports['iata'] = airports['iata'].astype(str).str.strip().str.upper()
airports = airports[airports['iata'].str.match(r"^[A-Z]{3}$")]

# --- Routes ---
routes_url = "https://raw.githubusercontent.com/jpatokal/openflights/master/data/routes.dat"

routes = pd.read_csv(
    routes_url,
    header=None,
    names=["airline","airline_id","source_airport","source_airport_id",
           "dest_airport","dest_airport_id","codeshare","stops","equipment"],
    quotechar='"'
)

# Merge coordinates
routes = routes.merge(
    airports[['iata','lat','lon']],
    left_on='source_airport',
    right_on='iata',
    how='left'
).rename(columns={'lat':'source_lat','lon':'source_lon'}).drop(columns='iata')

routes = routes.merge(
    airports[['iata','lat','lon']],
    left_on='dest_airport',
    right_on='iata',
    how='left'
).rename(columns={'lat':'dest_lat','lon':'dest_lon'}).drop(columns='iata')

# Drop routes with missing coordinates
routes = routes.dropna(subset=['source_lat','source_lon','dest_lat','dest_lon']).reset_index(drop=True)

# ===========================
# 3) Distance & CO2 Calculator
# ===========================

DEFAULT_FACTOR_G_PER_PKM = 83.0  # g CO₂ per passenger-km

def haversine_vec(lat1, lon1, lat2, lon2):
    """Vectorized great-circle distance in km"""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

routes['distance_km'] = haversine_vec(
    routes['source_lat'], routes['source_lon'],
    routes['dest_lat'], routes['dest_lon']
)

routes['co2_kg'] = routes['distance_km'] * DEFAULT_FACTOR_G_PER_PKM / 1000

# ===========================
# 4) BTS Arrival Delay Probabilities
# ===========================

bts_df = pd.read_csv(r"C:\Users\gregr\sustainable-flight-advisor\data\airline_delay_cause.csv")
bts_df['delay_prob'] = bts_df['arr_del15'] / bts_df['arr_flights']
bts_df = bts_df[['airport','carrier','delay_prob']]

# Merge with routes on destination airport + airline
routes_demo = routes.merge(
    bts_df,
    left_on=['dest_airport','airline'],
    right_on=['airport','carrier'],
    how='left'
)

routes_demo['delay_prob'] = routes_demo['delay_prob'].fillna(0.1)
routes_demo = routes_demo.drop(columns=['airport','carrier'])

# ===========================
# 5) Filter for active carriers using T-100 Domestic Segment
# ===========================

# --- Load T-100 Domestic CSV ---
t100 = pd.read_csv(r"C:\Users\gregr\sustainable-flight-advisor\data\t100_domestic.csv")

# Keep only the columns needed for active routes
active_routes = t100[['UNIQUE_CARRIER','ORIGIN','DEST']].drop_duplicates()

# Rename columns to match routes_demo
active_routes = active_routes.rename(columns={
    'UNIQUE_CARRIER':'airline',
    'ORIGIN':'source_airport',
    'DEST':'dest_airport'
})

# Filter routes_demo to only active routes
routes_demo = routes_demo.merge(
    active_routes,
    on=['airline','source_airport','dest_airport'],
    how='inner'
)

# ===========================
# 6) Streamlit UI
# ===========================

st.title("Sustainable Flight Advisor MVP")

origin_input = st.text_input("Origin IATA:", value="JFK").upper()
dest_input = st.text_input("Destination IATA:", value="SFO").upper()

# Collapse duplicates by airline + route first
routes_clean = routes_demo.groupby(
    ['airline','source_airport','dest_airport','distance_km','co2_kg'],
    as_index=False
).agg({'delay_prob':'mean'})

# Recompute score
routes_clean['score'] = routes_clean['co2_kg'] * 0.5 + routes_clean['delay_prob'] * 100

# Filter for user input
filtered = routes_clean[
    (routes_clean['source_airport']==origin_input) &
    (routes_clean['dest_airport']==dest_input)
].copy()

# Sort for display
filtered = filtered.sort_values('score')

st.write(f"Available routes from {origin_input} to {dest_input}:")
st.dataframe(filtered[['airline','source_airport','dest_airport','distance_km','co2_kg','delay_prob','score']])
