# syntax=docker/dockerfile:1

FROM nvidia/cuda:12.9.2-runtime-ubuntu24.04
WORKDIR /home/biolib/
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*
COPY checkpoints checkpoints
COPY pyproject.toml pyproject.toml
COPY deeptmhmm2_predictor deeptmhmm2_predictor
RUN pip install --no-cache-dir --retries 20 --timeout 300 --break-system-packages .

COPY sample.fasta sample.fasta
COPY biolib/README.md README.md
RUN dtm2 --model-dir checkpoints sample.fasta results

RUN mkdir output

ENTRYPOINT [""]