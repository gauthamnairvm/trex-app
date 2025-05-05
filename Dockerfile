FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-venv \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

RUN python3.10 -m pip install --upgrade pip
RUN python3.10 -m pip install -r requirements.txt

# Download NLTK data
RUN python3.10 -m nltk.downloader punkt stopwords

CMD ["python3.10", "main.py"]
