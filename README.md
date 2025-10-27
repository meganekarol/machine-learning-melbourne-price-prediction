# Deployment Guide (FastAPI)
This guide turns the finished scikit-learn pipeline into a simple, production-friendly HTTP API.

## 0) Prereqs
- Make sure you’re in the correct folder
- Ensure **custom transformers** (`FrequencyEncoder`, `DataFrameImputer`) are present in the util.py file
- Make sure **cleaned dataset** exists at `data/melbourne_housing_cleaned_featEngr.csv`
- Install the required dependencies ( `fastAPI` etc)
```bash
pip install -r requirements.txt

```

## 1) Export the fitted pipeline artifact
Loads the pipeline, fits on train+val (80%), and saves to `artifacts/`.

```bash
python export_model.py
# -> artifacts/model.joblib and artifacts/metadata.json
```
## 2) Run the app.py 
Expose the trained model as HTTP endpoints.
```bash
uvicorn app:app --reload
```


