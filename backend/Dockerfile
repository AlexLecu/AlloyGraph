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
# --timeout 300, not the default 30 or the previous 120. POST /api/validate
# runs the full Analyst -> Reviewer crew: two DeepInfra round-trips on a 70B
# model plus KG search. Measured 112 s for a 6-element wrought composition and
# >121 s for a 7-element one, which the old 120 s ceiling killed mid-LLM-call:
#   [CRITICAL] WORKER TIMEOUT (pid:8)  ->  500, worker SIGKILLed and replaced.
# The arbiter logs "Perhaps out of memory?" on that SIGKILL, which is
# misleading -- the host had 56 GiB available and zero OOM kills.
#
# This must stay >= the proxy_read_timeout in frontend/nginx.conf, otherwise
# gunicorn kills the request before nginx stops waiting for it.
#
# Note: these are sync workers, so one in-flight validate occupies a worker for
# up to 300 s; with --workers 4 that is four concurrent evaluations before
# requests queue.
CMD ["gunicorn", "--bind", "0.0.0.0:5001", "--workers", "4", "--timeout", "300", "--access-logfile", "-", "app:app"]
