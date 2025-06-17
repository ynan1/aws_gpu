cmp_test:
	nvcc test_cuda_exp.cu -o gpu_test -lpthread
cmp_exp_dbg:
	nvcc -g -pg -G exponent_cuda.cu -o gpu_exponent_dbg
cmp_exp:
	nvcc exponent_cuda.cu -o gpu_exponent
run_exp:
	./gpu_exponent;
clean:
	rm -rf gpu_exponent*;
	rm -rf gpu_test*;
