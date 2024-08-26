#!/bin/bash

INTERVAL=0.01

SUBSET="detail_23k.json"

N_SAMPLES=100
N_THREADS=4

echo "Evaluating on $SUBSET"
echo "Using $N_SAMPLES sample(s)."

mkdir results


variant="q2-matmul"
echo running $variant variant
mprof run --interval $INTERVAL --python -C -o results/${variant}_details.dat python eval.py -t $N_THREADS -n $N_SAMPLES -f moondream-${variant}.tflite --subset $SUBSET --out results/${variant}_details.csv


variant="q2-unpack"
echo running $variant variant
mprof run --interval $INTERVAL --python -C -o results/${variant}_details.dat python eval.py -t $N_THREADS -n $N_SAMPLES -f moondream-${variant}.tflite --subset $SUBSET --out results/${variant}_details.csv


variant="q2-tf"
echo running $variant variant
mprof run --interval $INTERVAL --python -C -o results/${variant}_details.dat python eval.py -t $N_THREADS -n $N_SAMPLES -f moondream-${variant}.tflite --subset $SUBSET --out results/${variant}_details.csv


variant="tf"
echo running $variant variant
mprof run --interval $INTERVAL --python -C -o results/${variant}_details.dat python eval.py -t $N_THREADS -n $N_SAMPLES -f moondream-${variant}.tflite --subset $SUBSET --out results/${variant}_details.csv


variant="tf-no-tf-quant"
echo running $variant variant
mprof run --interval $INTERVAL --python -C -o results/${variant}_details.dat python eval.py -t $N_THREADS -n $N_SAMPLES -f moondream-${variant}.tflite --subset $SUBSET --out results/${variant}_details.csv


variant="base"
echo running $variant variant
mprof run --interval $INTERVAL --python -C -o results/${variant}_details.dat python eval.py -t $N_THREADS -n $N_SAMPLES -c base --subset $SUBSET --out results/${variant}_details.csv


variant="binary_pt"
echo running $variant variant
mprof run --interval $INTERVAL --python -C -o results/${variant}_details.dat python eval.py -t $N_THREADS -n $N_SAMPLES -c checkpoints/moondream-11-1-1/model.safetensors --subset $SUBSET --out results/${variant}_details.csv


variant="ternary_pt"
echo running $variant variant
mprof run --interval $INTERVAL --python -C -o results/${variant}_details.dat python eval.py -t $N_THREADS -n $N_SAMPLES -c checkpoints/moondream-q2-1-1/model.safetensors --subset $SUBSET --out results/${variant}_details.csv