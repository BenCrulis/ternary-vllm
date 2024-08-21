#!/bin/bash

TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )


TF_HOME="/home/crulis/Documents/projects/tensorflow"

FLATBUFFERS_HOME="/home/crulis/Documents/projects/flatbuffers"

# TFLITE_HOME="/home/crulis/Documents/libs/tflite-dist/"
TFLITE_HOME="$TF_HOME"

# TFLITE_LFLAGS="-I$TF_HOME -ltensorflow"
TFLITE_LFLAGS="-I$TFLITE_HOME/include -I$TF_HOME/ -I$FLATBUFFERS_HOME/include -L$TF_HOME/bazel-bin/tensorflow/lite/c -ltensorflowlite_c "

echo "TFLITE_LFLAGS = " $TFLITE_LFLAGS

CXX=g++
NVCC=nvcc
PYTHON_BIN_PATH=python

CUSTOM_OPS_SRCS="native/tensorflow_custom_ops/cc/kernels/*.cc native/tensorflow_custom_ops/cc/ops/*.cc"

echo "Tensorflow CFLAGS: " $TF_CFLAGS
echo "Tensorflow LFLAGS: " $TF_LFLAGS

CC_PATH="native/tensorflow_zero_out/cc/ops/zero_out_ops.cc"
CC_KERNEL_PATH="native/tensorflow_zero_out/cc/kernels/zero_out_kernels.cc"

echo compiling ops
g++ -std=c++17 -shared -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} $TFLITE_LFLAGS -O2 -o custom_ops.so $CUSTOM_OPS_SRCS # $CC_KERNEL_PATH $CC_PATH


RESULT=$?
if [ $RESULT -eq 0 ]; then
  echo success
else
  echo failed
fi