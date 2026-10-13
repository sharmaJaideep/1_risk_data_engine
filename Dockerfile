# Risk Data Engine: one image for every pipeline step (`docker run <image> <step>`).
# NOTE: written but not built or run in development (no Docker available); treat the first
# build as a test.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 RISK_DATA_DIR=/data
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY main.py ./
COPY src ./src
COPY config ./config
COPY scripts ./scripts

# Data lives outside the image: mount a volume (or bucket-backed disk) at /data with raw/ inside it.
RUN useradd --create-home runner && mkdir -p /data && chown runner /data
USER runner
VOLUME /data

ENTRYPOINT ["python", "main.py"]
CMD ["--help"]
