#!/bin/bash

INTERVAL=0.01

N_SAMPLES=30
N_TOKENS=50
N_THREADS=4

OUTPUT_DIR="results_benchmarks"

echo "Using $N_SAMPLES sample(s)."

mkdir $OUTPUT_DIR

echo "Writing results to $OUTPUT_DIR"


variant="q2-matmul"
echo running $variant variant
mprof run --interval $INTERVAL --python -C -o $OUTPUT_DIR/${variant}.dat python benchmark.py -t $N_THREADS -n $N_SAMPLES -k $N_TOKENS -f moondream-${variant}.tflite --out $OUTPUT_DIR/${variant}.csv


variant="q2-unpack"
echo running $variant variant
mprof run --interval $INTERVAL --python -C -o $OUTPUT_DIR/${variant}.dat python benchmark.py -t $N_THREADS -n $N_SAMPLES -k $N_TOKENS -f moondream-${variant}.tflite --out $OUTPUT_DIR/${variant}.csv


variant="q2-tf"
echo running $variant variant
mprof run --interval $INTERVAL --python -C -o $OUTPUT_DIR/${variant}.dat python benchmark.py -t $N_THREADS -n $N_SAMPLES -k $N_TOKENS -f moondream-${variant}.tflite --out $OUTPUT_DIR/${variant}.csv


variant="tf"
echo running $variant variant
mprof run --interval $INTERVAL --python -C -o $OUTPUT_DIR/${variant}.dat python benchmark.py -t $N_THREADS -n $N_SAMPLES -k $N_TOKENS -f moondream-${variant}.tflite --out $OUTPUT_DIR/${variant}.csv


variant="tf-no-tf-quant"
echo running $variant variant
mprof run --interval $INTERVAL --python -C -o $OUTPUT_DIR/${variant}.dat python benchmark.py -t $N_THREADS -n $N_SAMPLES -k $N_TOKENS -f moondream-${variant}.tflite --out $OUTPUT_DIR/${variant}.csv
