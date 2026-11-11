# Data

| File | Rows | Produced by | Description |
|------|------|-------------|-------------|
| `raw/melbourne_housing_raw.csv` | 34,857 | — | Melbourne residential sales, Feb 2016 – Mar 2018 (prices in **AUD**) |
| `processed/melbourne_housing_cleaned_v1.csv` | 3,895 | `notebooks/01_DataExploration.ipynb` | Random sample of 5,000 listings (seed 42); rows without a price dropped; missing values imputed |
| `processed/melbourne_housing_cleaned_featEngr.csv` | 3,895 | `notebooks/02_PreprocessingPipeline.ipynb` | Model-ready: log-transformed areas, `PropertyAge`, `Log_TotalArea`, `Is_CBD` |

## Source

Derived from the [Melbourne Housing Market](https://www.kaggle.com/datasets/anthonypino/melbourne-housing-market)
dataset on Kaggle (scraped from Domain.com.au). The version used here was provided as part of a
university course and contains an extra `ParkingArea` column and deliberately corrupted values
(e.g. `inf`, `missing`) for the cleaning exercise. See the Kaggle page for the dataset's license terms.

## Columns (raw)

`Suburb`, `Address`, `Rooms`, `Type` (h = house, u = unit, t = townhouse), `Method` (sale method),
`SellerG` (agency), `Date`, `Distance` (km to CBD), `Postcode`, `Bedroom`, `Bathroom`, `Car`,
`Landsize` (m²), `BuildingArea` (m²), `YearBuilt`, `CouncilArea`, `Latitude`, `Longtitude`
(spelling as in source), `Regionname`, `Propertycount` (properties in suburb), `ParkingArea`,
`Price` (AUD, target).
