# syntax=docker/dockerfile:1

FROM nvidia/cuda:12.9.2-runtime-ubuntu24.04
WORKDIR /home/biolib/
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*
COPY checkpoints checkpoints
COPY pyproject.toml pyproject.toml
COPY deeptmhmm2_predictor deeptmhmm2_predictor
RUN pip install --no-cache-dir --retries 20 --timeout 300 --break-system-packages . \
    && pip install --no-cache-dir pandas tabulate

COPY sample.fasta sample.fasta
COPY biolib/README.md README.md
COPY biolib/make_report.py make_report.py
COPY biolib/run.sh run.sh
RUN chmod +x run.sh
RUN dtm2 --model-dir checkpoints sample.fasta results && python3 make_report.py

RUN mkdir output

ENTRYPOINT [""]