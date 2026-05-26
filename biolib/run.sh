#!/bin/bash
dtm2 --model-dir checkpoints "$@"
python3 make_report.py
