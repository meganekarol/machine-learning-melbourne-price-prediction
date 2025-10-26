# Base lightweight Python image
FROM python:3.11-slim

# Container working directory
WORKDIR /app

# (Optional) Install extra system libs here if needed, e.g., for XGBoost
# RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Copy your API code# Copy your API code
COPY api/ ./api/
# Copy your model + metadata
COPY artifacts/ ./artifacts/
# Include export script for traceability
COPY export_model.py ./

# Install dependencies
RUN pip install --no-cache-dir -r api/requirements.txt.txt

# Inform Docker the API runs on port 8000
EXPOSE 8000

# Run the FastAPI server
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
