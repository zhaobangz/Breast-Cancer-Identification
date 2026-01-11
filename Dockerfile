FROM python:3.10-slim

# set working directory
WORKDIR /app

# Install system dependencies required for some Python packages (OpenCV, build tools)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        libsm6 \
        libxext6 \
        libxrender-dev \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

# Copy project files
COPY . /app

ENV PYTHONUNBUFFERED=1

# Default to an interactive shell; run your training script with
# `docker run --rm -it <image> python train_model.py` or override CMD below
CMD ["bash"]
