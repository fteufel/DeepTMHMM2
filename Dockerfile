# syntax=docker/dockerfile:1

FROM python:3.14-slim
WORKDIR /home/biolib/

COPY checkpoints checkpoints
COPY pyproject.toml pyproject.toml
COPY deeptmhmm2_predictor deeptmhmm2_predictor
RUN pip install --no-cache-dir --retries 20 --timeout 300 .

COPY sample.fasta sample.fasta
RUN dtm2 --model-dir checkpoints sample.fasta results

RUN mkdir output
