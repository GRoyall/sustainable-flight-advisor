import streamlit as st
import pandas as pd
import joblib
from utils.data_loader import load_airports
from utils.emissions import estimate_route_emissions
from utils.scoring import compute_score

st.title("✈️ Sustainable Flight Advisor 🌱")
st.markdown("Estimate emissions and reliability between any two airports.")

# Load airport data
airports = load_airports()

# UI inputs
origins = airports["iata"].dropna().unique().tolist()
dests = airports["iata"].dropna().unique().tolist()
origin = st.selectbox("Origin Airport (IATA)", origins, index=origins.index("JFK") if "JFK" in origins else 0)
dest = st.selectbox("Destination Airport (IATA)", dests, index=dests.index("LAX") if "LAX" in dests else 0)

# Dummy model (for now)
try:
    model = joblib.load("model/model.pkl")
except:
    model = None

if st.button("Calculate Flight Impact"):
    co2_data = estimate_route_emissions(airports, origin, dest)
    if not co2_data:
        st.error("Airport not found.")
    else:
        distance, co2_kg = co2_data
        st.metric("Distance (km)", f"{distance:.0f}")
        st.metric("Estimated CO₂ (kg)", f"{co2_kg:.1f}")

        # placeholder delay probability and duration
        delay_prob = 0.25
        duration_hr = distance / 800  # assume avg 800 km/h
        score = compute_score(delay_prob, co2_kg, duration_hr)

        st.success(f"Composite Sustainability Score: {score}")
        st.caption("Higher is better (balances reliability, emissions, and travel time)")
