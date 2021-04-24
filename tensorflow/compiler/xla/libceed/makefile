TENSORFLOW_BASE := /home/emily/tensorflow/bazel-tensorflow
TENSORFLOW_DIR := /home/emily/tensorflow
TENSORFLOW_LIB := $(TENSORFLOW_BASE)/bazel-bin/tensorflow/libtensorflow.so
ABSEIL_BASE := /home/emily/abseil-cpp
EIGEN_BASE := /home/emily/eigen
INCLS := -I$(TENSORFLOW_BASE) -I$(ABSEIL_BASE) -I$(EIGEN_BASE) -I$(TENSORFLOW_DIR) -I$(TENSORFLOW_BASE)/tensorflow/core/framework
LDFLAGS := -L$(TENSORFLOW_LIB) -ltensorflow

CXX := clang++
CXXOPT := -O3 -g -ffp-contract=fast
CXXFLAGS = -std=c++17 $(INCLS) $(CXXOPT) -shared -fPIC

LIB_TARGET := libxla_caller.so

SRCS := xla_caller.cc\

.PHONY: $(LIB_TARGET)

default: all

all: $(LIB_TARGET)

$(LIB_TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $^ -o $@
