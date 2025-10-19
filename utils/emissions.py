import math
import pandas as pd

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return R * 2 * math.asin(math.sqrt(a))

def estimate_co2_kg(lat1, lon1, lat2, lon2, factor_g_per_pkm=83.0):
    distance_km = haversine_km(lat1, lon1, lat2, lon2)
    return distance_km, (distance_km * factor_g_per_pkm) / 1000  # kg CO₂

def estimate_route_emissions(airports_df, origin, dest):
    a1 = airports_df.loc[airports_df['iata'] == origin]
    a2 = airports_df.loc[airports_df['iata'] == dest]
    if a1.empty or a2.empty:
        return None
    lat1, lon1 = a1.iloc[0]['lat'], a1.iloc[0]['lon']
    lat2, lon2 = a2.iloc[0]['lat'], a2.iloc[0]['lon']
    return estimate_co2_kg(lat1, lon1, lat2, lon2)
