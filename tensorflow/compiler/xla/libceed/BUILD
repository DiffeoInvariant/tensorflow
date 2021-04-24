#package(
#    default_visibility = ["//package:external"],
#)
load("//tensorflow:tensorflow.bzl", "tf_cc_binary", "tf_cc_shared_object")

tf_cc_binary(
	name = "libxla_caller.so",
	srcs = ["xla_caller.hpp", "xla_caller.cc"],
	deps = [
        "//tensorflow/compiler/xla/tools:hlo_module_loader",
      "//tensorflow/compiler/xla/pjrt:pjrt_client",
      "//tensorflow/compiler/xla/pjrt:cpu_device",
      "//tensorflow/compiler/jit:xla_cpu_jit",
      "//tensorflow/compiler/xla:status",
      "//tensorflow/compiler/xla:statusor",
      "//tensorflow/compiler/xla:literal",
      "//tensorflow/compiler/xla:literal_util",
      "//tensorflow/compiler/xla:shape_util",
      "//tensorflow/compiler/xla/service:hlo_proto_cc",
    ],
	linkshared = 1,
)

