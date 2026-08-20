FROM python:3.11-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Build the optional C++ bar loop (Phase 7). build-essential is already
# installed above. The engine falls back to pure Python if this is absent,
# but the image ships with it so `docker run` exercises the fast path.
RUN python setup.py build_ext --inplace

RUN mkdir -p data/cache notebooks/results

CMD ["python", "-m", "engine.run"]
