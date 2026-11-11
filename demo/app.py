"""Streamlit demo: estimate the sale price of a Melbourne property.

Run locally:
    streamlit run demo/app.py
"""

import math
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Make the package importable without `pip install` (e.g. on Streamlit Community Cloud).
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from melbourne_price.predict import load_metadata, predict_prices  # noqa: E402

CLEANED_DATA = ROOT / "data" / "processed" / "melbourne_housing_cleaned_v1.csv"
TYPE_LABELS = {"h": "House", "u": "Unit / apartment", "t": "Townhouse"}

st.set_page_config(page_title="Melbourne House Price Estimator", page_icon="🏠", layout="wide")


@st.cache_data
def load_sales() -> pd.DataFrame:
    return pd.read_csv(CLEANED_DATA)


@st.cache_data
def suburb_profiles(sales: pd.DataFrame) -> pd.DataFrame:
    """Typical location attributes per suburb, used to pre-fill the form."""

    def mode(s: pd.Series):
        return s.mode().iloc[0]

    profiles = sales.groupby("Suburb").agg(
        CouncilArea=("CouncilArea", mode),
        Regionname=("Regionname", mode),
        Distance=("Distance", "median"),
        Latitude=("Latitude", "median"),
        Longtitude=("Longtitude", "median"),
        Propertycount=("Propertycount", "median"),
        Sales=("Price", "size"),
    )
    # ~22% of rows had missing coordinates filled with one global mean during cleaning.
    # Use only real coordinates for each suburb's location where available.
    coords = ["Latitude", "Longtitude"]
    imp_lat, imp_lon = sales.groupby(coords).size().idxmax()
    real = sales[(sales["Latitude"] != imp_lat) | (sales["Longtitude"] != imp_lon)]
    profiles.update(real.groupby("Suburb")[coords].median())
    return profiles


sales = load_sales()
profiles = suburb_profiles(sales)
meta = load_metadata()
mae_log = meta.get("test_metrics", {}).get("mae_log", 0.18)

st.title("🏠 Melbourne House Price Estimator")
st.caption(
    "Gradient Boosting model trained on 2016–2018 Melbourne sales. "
    f"Test R² {meta.get('test_metrics', {}).get('r2_log', 0):.2f} (log price) · "
    "[Source code](https://github.com/meganekarol/machine-learning-melbourne-price-prediction)"
)

form_col, result_col = st.columns([1, 1], gap="large")

with form_col:
    st.subheader("Property details")
    suburbs = profiles.index.tolist()
    suburb = st.selectbox("Suburb", suburbs, index=suburbs.index("Richmond"))
    profile = profiles.loc[suburb]

    prop_type = st.radio(
        "Property type", list(TYPE_LABELS), format_func=TYPE_LABELS.get, horizontal=True
    )
    c1, c2, c3 = st.columns(3)
    bedrooms = c1.number_input("Bedrooms", 0, 10, 3)
    bathrooms = c2.number_input("Bathrooms", 0, 8, 2)
    car = c3.number_input("Car spaces", 0, 10, 1)

    c4, c5 = st.columns(2)
    land = c4.number_input("Land size (m²)", 0, 5000, 450, step=10)
    building = c5.number_input("Building area (m²)", 10, 1500, 140, step=5)
    year_built = st.slider("Year built", 1850, 2017, 1970)

    with st.expander("Advanced: location details (pre-filled from suburb)"):
        distance = st.number_input(
            "Distance to CBD (km)", 0.0, 60.0, float(profile.Distance), step=0.1
        )
        council = st.text_input("Council area", profile.CouncilArea)
        region = st.text_input("Region", profile.Regionname)
        lat = st.number_input("Latitude", -39.0, -37.0, float(profile.Latitude), format="%.4f")
        lon = st.number_input("Longitude", 144.0, 146.0, float(profile.Longtitude), format="%.4f")

listing = pd.DataFrame(
    [
        {
            "Suburb": suburb,
            "Type": prop_type,
            "Distance": distance,
            "Bedroom": bedrooms,
            "Bathroom": bathrooms,
            "Car": car,
            "Landsize": land,
            "BuildingArea": building,
            "YearBuilt": year_built,
            "CouncilArea": council,
            "Regionname": region,
            "Latitude": lat,
            "Longtitude": lon,
            "Propertycount": profile.Propertycount,
        }
    ]
)
price = float(predict_prices(listing)[0])
low, high = price * math.exp(-mae_log), price * math.exp(mae_log)

with result_col:
    st.subheader("Estimated sale price")
    st.metric("Estimate (AUD)", f"${price:,.0f}")
    st.write(
        f"Typical range: **${low:,.0f} – ${high:,.0f}** (± the model's average test-set error)."
    )

    comps = sales[(sales.Suburb == suburb) & (sales.Type == prop_type)]
    if len(comps):
        st.write(
            f"For context: **{len(comps)}** {TYPE_LABELS[prop_type].lower()} sale(s) in "
            f"{suburb} in the data, median **${comps.Price.median():,.0f}**."
        )
    else:
        st.write(f"No {TYPE_LABELS[prop_type].lower()} sales in {suburb} in the training data.")

    st.map(pd.DataFrame({"lat": [lat], "lon": [lon]}), zoom=11)

st.divider()
st.caption(
    "Estimates reflect 2016–2018 market conditions and are for demonstration only. "
    "Not financial advice."
)
