"""FastAPI service for Melbourne house price predictions.

Run locally:
    uvicorn api.main:app --reload
Then open http://127.0.0.1:8000/docs
"""

from contextlib import asynccontextmanager
from typing import Annotated, Literal

import pandas as pd
from fastapi import Body, FastAPI
from pydantic import BaseModel, ConfigDict, Field

from melbourne_price import __version__
from melbourne_price.predict import load_metadata, load_model, predict_prices

EXAMPLE_LISTING = {
    "suburb": "Northcote",
    "property_type": "h",
    "region": "Northern Metropolitan",
    "council_area": "Darebin City Council",
    "distance_km": 5.3,
    "bedrooms": 3,
    "bathrooms": 1,
    "car_spaces": 2,
    "land_size_m2": 515,
    "building_area_m2": 148,
    "year_built": 1960,
    "latitude": -37.8065,
    "longitude": 144.9927,
    "sale_method": "S",
    "seller": "Jellis",
    "suburb_property_count": 11364,
}

# API field name -> column name in the training data
FIELD_TO_COLUMN = {
    "suburb": "Suburb",
    "property_type": "Type",
    "region": "Regionname",
    "council_area": "CouncilArea",
    "distance_km": "Distance",
    "bedrooms": "Bedroom",
    "bathrooms": "Bathroom",
    "car_spaces": "Car",
    "land_size_m2": "Landsize",
    "building_area_m2": "BuildingArea",
    "year_built": "YearBuilt",
    "latitude": "Latitude",
    "longitude": "Longtitude",
    "sale_method": "Method",
    "seller": "SellerG",
    "suburb_property_count": "Propertycount",
}


class Listing(BaseModel):
    """A residential property listing in Greater Melbourne."""

    model_config = ConfigDict(json_schema_extra={"examples": [EXAMPLE_LISTING]})

    suburb: str = Field(description="Suburb name, e.g. 'Richmond'")
    property_type: Literal["h", "u", "t"] = Field(description="h = house, u = unit, t = townhouse")
    region: str = Field(description="Metropolitan region, e.g. 'Southern Metropolitan'")
    council_area: str = Field(description="Local council, e.g. 'Yarra City Council'")
    distance_km: float = Field(ge=0, le=100, description="Distance to the Melbourne CBD in km")
    bedrooms: int = Field(ge=0, le=20)
    bathrooms: int = Field(ge=0, le=20)
    car_spaces: int = Field(ge=0, le=20)
    land_size_m2: float = Field(ge=0, description="Land size in square metres")
    building_area_m2: float = Field(ge=0, description="Building area in square metres")
    year_built: int = Field(ge=1800, le=2030)
    latitude: float = Field(ge=-39, le=-37)
    longitude: float = Field(ge=144, le=146)
    sale_method: Literal["S", "SP", "PI", "VB", "SA"] | None = Field(
        default=None, description="Optional. S = sold, SP = sold prior, PI = passed in, ..."
    )
    seller: str | None = Field(default=None, description="Optional. Real estate agency")
    suburb_property_count: float | None = Field(
        default=None, ge=0, description="Optional. Number of properties in the suburb"
    )


class Prediction(BaseModel):
    price_aud: float


class PredictionResponse(BaseModel):
    predictions: list[Prediction]
    currency: Literal["AUD"] = "AUD"
    model_version: str


def listings_to_frame(listings: list[Listing]) -> pd.DataFrame:
    rows = [listing.model_dump() for listing in listings]
    return pd.DataFrame(rows).rename(columns=FIELD_TO_COLUMN)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()  # fail fast at startup if the model is missing
    yield


app = FastAPI(
    title="Melbourne Housing Price Predictor",
    description="Predicts the sale price (AUD) of a Melbourne property "
    "with a tuned Gradient Boosting model.",
    version=__version__,
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/version")
def version() -> dict:
    meta = load_metadata()
    return {
        "app_version": __version__,
        "model_type": meta.get("model_type"),
        "trained_at": meta.get("created_at"),
        "test_metrics": meta.get("test_metrics"),
    }


def _respond(listings: list[Listing]) -> PredictionResponse:
    prices = predict_prices(listings_to_frame(listings))
    return PredictionResponse(
        predictions=[Prediction(price_aud=round(float(p), 2)) for p in prices],
        model_version=__version__,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(
    listings: Annotated[list[Listing], Body(min_length=1, max_length=1000)],
) -> PredictionResponse:
    """Predict prices for a batch of listings (max 1000)."""
    return _respond(listings)


@app.post("/predict_one", response_model=PredictionResponse)
def predict_one(listing: Listing) -> PredictionResponse:
    """Predict the price of a single listing."""
    return _respond([listing])
