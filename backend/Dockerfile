# 3.12 to match requirements.lock, which is compiled for --python-version 3.12
# and pins the scikit-learn / xgboost / numpy that saved_models_v2 was trained
# on (alloy_crew/models/train_env.json). Serving those pickles under a
# different stack is exactly the drift the lock exists to prevent.
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
  build-essential \
  curl \
  && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies from the fully pinned lock, not the
# open-ended requirements.txt -- a rebuild must not silently pick up a newer
# scikit-learn than the one saved_models_v2 was pickled with.
COPY requirements.lock .
RUN pip install --no-cache-dir --upgrade pip && \
  pip install --no-cache-dir -r requirements.lock

# Copy application code
COPY . .

# Expose Flask port
EXPOSE 5001

# Run with Gunicorn (production WSGI server).
#
# --worker-class gthread, 4 workers x 8 threads = 32 concurrent requests.
#
# Previously 4 *sync* workers, i.e. 4 concurrent requests total. Any four
# slow operations -- chat streams most of all, since one holds its worker for
# the whole 13-30 s response -- blocked everything else, INCLUDING /health.
# A health probe that queues behind real work reports the container unhealthy
# while it is serving perfectly well; that is the same false-negative class as
# the frontend healthcheck bug fixed earlier, and it makes the signal useless
# exactly when you need it.
#
# gthread rather than more sync workers, because:
#   * The work is I/O-bound -- HTTPS to DeepInfra, gRPC/HTTP to Weaviate. The
#     GIL is released during those waits, which is precisely where threads win.
#   * Threads inside a worker SHARE the loaded XGBoost/sklearn models. Each
#     sync worker lazily loads its own copy after fork (measured: a warm worker
#     is ~560 MB against ~285 MB cold). Scaling concurrency with processes
#     would duplicate ~275 MB of identical read-only model per worker; 32 sync
#     workers would cost ~18 GB to do what 4 gthread workers do in ~2.2 GB.
#   * Background job threads already run outside gunicorn's pool, so the app is
#     multi-threaded in production today. gthread widens an existing exposure
#     rather than opening a new one -- and the two check-then-act singleton
#     races that exposure implies are fixed in this same change.
#   * 4 worker processes still give real CPU parallelism for the short
#     XGBoost inference bursts, across 16 available cores.
#
# Deliberately NOT using --preload. It would load the models once before fork
# and share them copy-on-write, but with 56 GiB free the memory is not needed,
# and preloading trades that for fork-safety risk around anything holding a
# socket or a native thread pool at import time.
#
# --timeout 300 is kept for the same reason as before (it must stay >= the
# proxy_read_timeout in frontend/nginx.conf), though it now matters far less:
# every long operation runs as a background job, so requests are short.
# Note gthread heartbeats from the worker's main loop, so a slow request no
# longer risks the SIGABRT-and-replace behaviour sync workers had.
CMD ["gunicorn", "--bind", "0.0.0.0:5001", \
  "--worker-class", "gthread", "--workers", "4", "--threads", "8", \
  "--timeout", "300", "--graceful-timeout", "30", \
  "--access-logfile", "-", "app:app"]
