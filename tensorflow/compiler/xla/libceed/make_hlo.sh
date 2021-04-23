#!/bin/sh

env JAX_ENABLE_X64=True python jax_to_hlo.py --fn example_program.cr_j --input_shapes '[("x", "f64[3,3]"), ("y", "f64[3,3]")]' --hlo_text_dest cr_jac_hlo.txt --hlo_proto_dest cr_jac_hlo.pb


env JAX_ENABLE_X64=True python jax_to_hlo.py --fn example_program.cr_fn --input_shapes '[("x", "f64[3,3]"), ("y", "f64[3,3]")]' --hlo_text_dest cr_fn_hlo.txt --hlo_proto_dest cr_fn_hlo.pb
