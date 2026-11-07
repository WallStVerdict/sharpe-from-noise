# sharpe-from-noise: reproduce every number and chart in a clean container.
#   docker build -t sharpe-from-noise . && docker run --rm -v "$PWD/results:/app/results" sharpe-from-noise
# Parts 1 and 2 need no data. real_market.py downloads the Ken French file when the container runs (needs network).
FROM python:3.12-slim
WORKDIR /app
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends make && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY Makefile noise_sharpe.py walk_forward.py real_market.py data.sha256 ./
ENV MPLCONFIGDIR=/tmp/mpl IN_DOCKER=1
CMD ["make", "reproduce"]
