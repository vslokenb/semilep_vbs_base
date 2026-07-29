#!/bin/bash
set -e

INPUT_DIR=$1

echo "Running mix_years.py on: $INPUT_DIR"
echo "Host: $(hostname), Time: $(date)"

python mix_years.py "$INPUT_DIR"