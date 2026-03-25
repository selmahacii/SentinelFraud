# =============================================================================
# SENTINEL-FRAUD: Kafka Producer Dockerfile
# =============================================================================

FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY producers/ ./producers/

# Run producer
CMD ["python", "-m", "producers.transaction_producer"]
