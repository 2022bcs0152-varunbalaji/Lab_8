FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
    pandas==2.0.3 \
    scikit-learn==1.3.0 \
    dvc[s3]==3.30.0

COPY . .

CMD ["python", "app/train.py"]
