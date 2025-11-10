FROM python:3.12-slim

WORKDIR /app

# Install gsutil for downloading from GCS
RUN apt-get update && \
    apt-get install -y curl gnupg && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - && \
    apt-get update && apt-get install -y google-cloud-cli && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Download data from GCS during build
RUN mkdir -p data/geo models && \
    gsutil cp gs://team-53-data/geo/zcta_3857.parquet data/geo/zcta_3857.parquet && \
    gsutil cp gs://team-53-data/models/pipeline.joblib models/pipeline.joblib && \
    gsutil cp gs://team-53-data/models/infer_meta.json models/infer_meta.json

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]