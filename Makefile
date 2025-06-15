compile_exponent_debug:
	nvcc -g -G exponent_cuda.cu -o gpu_exponent_dbg
compile_exponent:
	nvcc exponent_cuda.cu -o gpu_exponent
run_exponent:
	./gpu_exponent
