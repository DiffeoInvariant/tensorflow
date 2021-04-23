TENSORFLOW_BASE := /home/emily/tensorflow
TENSORFLOW_LIB := $(TENSORFLOW_BASE)/TODO-FILL-IN-HERE
ABSEIL_BASE := /home/emily/abseil-cpp
INCLS := -I$(TENSORFLOW_BASE)/tensorflow -I$(ABSEIL_BASE)
LDFLAGS := -L$(TENSORFLOW_LIB) -ltensorflow

CXX := clang++
CXXOPT := -O3 -g -ffp-contract=fast
CXXFLAGS = -std=c++17 $(INCLS) $(CXXOPT) -shared -fPIC

LIB_TARGET := libxla_caller.so

SRCS := xla_caller.cc

all: $(LIB_TARGET)

$(LIB_TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $^ -o $@
