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

.PHONY: cmp_exp cmp_exp_dbg run_exp clean cmp_test cmp_test_dbg run_test cmp_test_unified cmp_test_unified_dbg softmax_fused softmax_fused_dbg softmax_test softmax_test_dbg
softmax_fused_1sc_v2:
	nvcc -Iinclude main_single_class.cu src/softmax_fused_1sc_v2.cu -o gpu_softmax_fused_1sc_v2
softmax_fused_1sc_v2_dbg:
	nvcc -g -G -Iinclude main_single_class.cu src/softmax_fused_1sc_v2.cu -o gpu_softmax_fused_1sc_v2_dbg
softmax_fused_1cl:
	nvcc -Iinclude main_single_class.cu src/softmax_fused_1sc.cu -o gpu_softmax_fused_1cl
softmax_fused_1cl_dbg:
	nvcc -g -G -Iinclude main_single_class.cu src/softmax_fused_1sc.cu -o gpu_softmax_fused_1cl_dbg
softmax_fused_new:
	nvcc -Iinclude main.cu src/softmax_fused_opt.cu -o gpu_softmax_fused_new
softmax_fused_new_dbg:
	nvcc -g -G -Iinclude main.cu src/softmax_fused_opt.cu -o gpu_softmax_fused_new_dbg
softmax_3stream:
	nvcc -Iinclude src/softmax_3stream.cu -o gpu_softmax_3stream
softmax_3stream_dbg:
	nvcc -g -G -Iinclude src/softmax_3stream.cu -o gpu_softmax_3stream_dbg
softmax_fused:
	nvcc -I include main.cu src/softmax_fused.cu -o gpu_softmax_fused
softmax_fused_dbg:
	nvcc -g -G -Iinclude main.cu src/softmax_fused.cu -o gpu_softmax_fused_dbg
softmax_test:
	nvcc -I include src/softmax_1stream.cu -o gpu_softmax_test
softmax_test_dbg:
	nvcc -g -G -Iinclude src/softmax_1stream.cu -o gpu_softmax_test_dbg
cmp_test_unified:
	nvcc -DUSE_MANAGED -Iinclude src/test_cuda_exp.cu -o gpu_test_unified
cmp_test_unified_dbg:
	nvcc -g -G -DUSE_MANAGED -Iinclude src/test_cuda_exp.cu -o gpu_test_unified_dbg
cmp_test:
	nvcc -Iinclude src/test_cuda_exp.cu -o gpu_test
cmp_test_dbg:
	nvcc -g -G -Iinclude src/test_cuda_exp.cu -o gpu_test_dbg
run_test:
	./gpu_test;
cmp_exp_dbg:
	nvcc -g -G -Iinclude src/exponent_cuda.cu -o gpu_exponent_dbg
cmp_exp:
	nvcc -Iinclude src/exponent_cuda.cu -o gpu_exponent
run_exp:
	./gpu_exponent;
clean:
	rm -rf gpu_exponent*;
	rm -rf gpu_test*;
	rm -rf gpu_softmax_*;

