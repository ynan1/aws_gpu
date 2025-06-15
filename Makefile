compile_exponent:
	nvcc -g -G exponent_cuda.cu -o gpu_exponent
run_exponent:
	./gpu_exponent
