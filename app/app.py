# ===========================
# Sustainable Flight Advisor MVP (AI + Aircraft CO₂ + Insights)
# ===========================

# --- 1) Imports ---
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ===========================
# 2) Load Airports & Routes
# ===========================
airports_url = "https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat"
colnames = ["id","name","city","country","iata","icao","lat","lon","alt","tz","dst","tzdb","type","source"]

airports = pd.read_csv(airports_url, header=None, names=colnames, quotechar='"', skipinitialspace=True)
airports['iata'] = airports['iata'].astype(str).str.strip().str.upper()
airports = airports[airports['iata'].str.match(r"^[A-Z]{3}$")]

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
    left_on='source_airport', right_on='iata', how='left'
).rename(columns={'lat':'source_lat','lon':'source_lon'}).drop(columns='iata')

routes = routes.merge(
    airports[['iata','lat','lon']],
    left_on='dest_airport', right_on='iata', how='left'
).rename(columns={'lat':'dest_lat','lon':'dest_lon'}).drop(columns='iata')

routes = routes.dropna(subset=['source_lat','source_lon','dest_lat','dest_lon']).reset_index(drop=True)

# ===========================
# 3) Distance & Base CO₂ Calculator
# ===========================
DEFAULT_FACTOR_G_PER_PKM = 83.0

def haversine_vec(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

routes['distance_km'] = haversine_vec(routes['source_lat'], routes['source_lon'], routes['dest_lat'], routes['dest_lon'])
routes['co2_kg_base'] = routes['distance_km'] * DEFAULT_FACTOR_G_PER_PKM / 1000

# ===========================
# 4) BTS Arrival Delay Probabilities
# ===========================

from pathlib import Path
base_dir = Path(__file__).resolve().parents[1]  # repo root
bts_df = pd.read_csv(base_dir / "data" / "Airline_Delay_Cause.csv")
#bts_df = pd.read_csv("../data/Airline_Delay_Cause.csv")

bts_df['delay_prob'] = bts_df['arr_del15'] / bts_df['arr_flights']
bts_df = bts_df[['airport', 'carrier', 'delay_prob']]

routes_demo = routes.merge(
    bts_df,
    left_on=['dest_airport','airline'],
    right_on=['airport','carrier'],
    how='left'
)
routes_demo['delay_prob'] = routes_demo['delay_prob'].fillna(0.1)
routes_demo = routes_demo.drop(columns=['airport','carrier'])

# ===========================
# 5) Aircraft CO₂ Factor
# ===========================

aircraft_ref = pd.read_csv(base_dir / "data" / "Aircraftlookup.csv")
aircraft_ref['Code_str'] = aircraft_ref['Code'].astype(str).str.zfill(3)
aircraft_co2_map = dict(zip(aircraft_ref['Code_str'], aircraft_ref['CO2']))

def map_aircraft_co2(equip_str):
    if pd.isna(equip_str) or equip_str.strip() == '':
        return np.nan
    codes = equip_str.strip().split()
    co2_vals = [aircraft_co2_map.get(c.zfill(3), np.nan) for c in codes]
    return np.nanmean(co2_vals)

routes_demo['aircraft_CO2_factor'] = routes_demo['equipment'].apply(map_aircraft_co2)

# Average aircraft CO₂ by (airline, source_airport)
carrier_city_avg = (
    routes_demo.groupby(['airline','source_airport'], as_index=False)['aircraft_CO2_factor']
    .mean()
    .rename(columns={'aircraft_CO2_factor':'avg_aircraft_CO2_factor'})
)

routes_demo = routes_demo.merge(carrier_city_avg, on=['airline','source_airport'], how='left')
routes_demo['avg_aircraft_CO2_factor'] = routes_demo['avg_aircraft_CO2_factor'].fillna(50)

routes_demo['co2_kg'] = routes_demo['co2_kg_base'] * (routes_demo['avg_aircraft_CO2_factor'] / 50)

# ===========================
# 6) Filter for Active Carriers
# ===========================


t100 = pd.read_csv(base_dir / "data" / "t100_domestic.csv")
active_routes = t100[['UNIQUE_CARRIER','ORIGIN','DEST']].drop_duplicates()
active_routes = active_routes.rename(columns={'UNIQUE_CARRIER':'airline','ORIGIN':'source_airport','DEST':'dest_airport'})

routes_demo = routes_demo.merge(active_routes, on=['airline','source_airport','dest_airport'], how='inner')

# ===========================
# 7) AI: Random Forest Model
# ===========================
routes_demo['delayed_binary'] = (routes_demo['delay_prob'] > 0.15).astype(int)
X = routes_demo[['airline','distance_km']]
y = routes_demo['delayed_binary']
X_encoded = pd.get_dummies(X, columns=['airline'])
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# ===========================
# 8) Streamlit UI
# ===========================
st.title("Sustainable Flight Advisor MVP")

# --- User Scenarios ---
fullness_option = st.select_slider(
    "How full do you expect flights to be when you travel?",
    options=["Less Full", "Full", "Very Full"],
    value="Full"
)
load_factor = {"Less Full":0.6, "Full":0.85, "Very Full":0.99}[fullness_option]

delay_threshold_min = st.slider(
    "How much of a delay is too long? (minutes)",
    min_value=0, max_value=160, value=15, step=5
)

delay_weight = st.slider("How important is on-time performance to you?", 0.0, 1.0, 0.5, 0.05)
co2_weight = st.slider("How important is sustainability to you?", 0.0, 1.0, 0.5, 0.05)

# --- Route Inputs ---
origin_input = st.text_input("Departure Airport Code:", value="JFK").upper()
dest_input = st.text_input("Arrival Airport Code:", value="SFO").upper()

# Collapse duplicates by airline + route
routes_clean = routes_demo.groupby(
    ['airline','source_airport','dest_airport','distance_km','co2_kg','avg_aircraft_CO2_factor'],
    as_index=False
).agg({'delay_prob':'mean'})

filtered = routes_clean[(routes_clean['source_airport']==origin_input) & (routes_clean['dest_airport']==dest_input)].copy()

if not filtered.empty:
    # AI delay prediction
    filtered_features = filtered[['airline','distance_km']]
    filtered_features_encoded = pd.get_dummies(filtered_features, columns=['airline'])
    filtered_features_encoded = filtered_features_encoded.reindex(columns=X_encoded.columns, fill_value=0)
    filtered['ai_delay_prob'] = rf_model.predict_proba(filtered_features_encoded)[:,1]

    # Adjust CO₂ per passenger using load factor
    filtered['co2_per_passenger_kg'] = filtered['co2_kg'] / load_factor

    # Weighted score
    filtered['score'] = filtered['co2_per_passenger_kg']*co2_weight + filtered['ai_delay_prob']*100*delay_weight
    filtered = filtered.sort_values('score')

# --- Results summary ---
if not filtered.empty:
    best_carrier = filtered.loc[filtered["score"].idxmin(), "airline"]
    avg_co2 = filtered["co2_per_passenger_kg"].mean()
    distance_km = int(filtered["distance_km"].iloc[0])

    st.markdown(
        f"**{origin_input} to {dest_input}** is about **{distance_km:,} km** "
        f"and emits roughly **{int(avg_co2):,} lbs of CO₂ per passenger.**  \n"
        f"✅ **Best carrier for you: {best_carrier}**"
    )

    # --- Results table ---
    display_df = filtered[["airline", "co2_per_passenger_kg", "delay_prob", "score"]].copy()
    display_df.columns = ["Carrier", "CO₂ (lbs)", "Delay Probability", "Score"]

    # Format and center values
    display_df["CO₂ (lbs)"] = display_df["CO₂ (lbs)"].map(lambda x: f"{int(x):,}")
    display_df["Delay Probability"] = display_df["Delay Probability"].map(lambda x: f"{x:.2f}")
    display_df["Score"] = display_df["Score"].map(lambda x: f"{x:.2f}")

    st.dataframe(
        display_df.style.set_properties(**{"text-align": "center"})
                         .set_table_styles(
                             [{"selector": "th", "props": [("text-align", "center")]}]
                         ),
        hide_index=True
    )

else:
    st.write(f"No active routes found from {origin_input} to {dest_input}.")
