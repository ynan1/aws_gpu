# Makefile for CUDA exponentiation example
# This Makefile compiles the CUDA code for exponentiation and runs tests.
# It includes options for debugging and profiling.
# Usage:
#   - `make cmp_exp_dbg` to compile with debugging symbols
#   - `make cmp_exp` to compile without debugging symbols
#   - `make run_exp` to run the compiled exponentiation program
#   - `make clean` to remove compiled files
#   - `make cmp_test` to compile the test program
#   - `make cmp_test_dbg` to compile the test program with debugging symbols
#   - `make run_test` to run the test program
#   - `make clean` to remove compiled test files

.PHONY: cmp_exp cmp_exp_dbg run_exp clean cmp_test cmp_test_dbg run_test

cmp_test_unified:
	nvcc -DUSE_MANAGED test_cuda_exp.cu -o gpu_test_unified
cmp_test_unified_dbg:
	nvcc -g -G -DUSE_MANAGED test_cuda_exp.cu -o gpu_test_unified_dbg
cmp_test:
	nvcc test_cuda_exp.cu -o gpu_test
cmp_test_dbg:
	nvcc -g -G test_cuda_exp.cu -o gpu_test_dbg
run_test:
	./gpu_test;
cmp_exp_dbg:
	nvcc -g -G exponent_cuda.cu -o gpu_exponent_dbg
cmp_exp:
	nvcc exponent_cuda.cu -o gpu_exponent
run_exp:
	./gpu_exponent;
clean:
	rm -rf gpu_exponent*;
	rm -rf gpu_test*;
