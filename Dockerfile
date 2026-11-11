# Melbourne house price API
#   docker build -t melbourne-price-api .
#   docker run -p 8000:8000 melbourne-price-api   ->  http://localhost:8000/docs
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MELBOURNE_MODELS_DIR=/app/models

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install ".[api]"

COPY api/ api/
COPY models/ models/

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
