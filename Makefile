cmp_exp_dbg:
	nvcc -g -pg -G exponent_cuda.cu -o gpu_exponent_dbg
cmp_exp:
	nvcc exponent_cuda.cu -o gpu_exponent
run_exp:
	./gpu_exponent;
clean:
	rm -rf gpu_exponent*;
