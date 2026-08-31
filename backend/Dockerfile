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

# Run with Gunicorn (production WSGI server)
CMD ["gunicorn", "--bind", "0.0.0.0:5001", "--workers", "4", "--timeout", "120", "--access-logfile", "-", "app:app"]
