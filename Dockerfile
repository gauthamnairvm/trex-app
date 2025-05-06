FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.10 python3-pip python3.10-venv python3-tk \
    git curl build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

COPY . /app

RUN pip install --upgrade pip && pip install -r requirements.txt

RUN pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0+cu118 --index-url https://download.pytorch.org/whl/cu118

RUN python -m nltk.downloader punkt stopwords

CMD ["python", "main.py"]
