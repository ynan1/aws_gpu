#include<iostream>
#include<ctime>
#include<cstdlib>
#include<cuda_runtime.h>

#define N 1000

using namespace std;
__global__ void exponent(float* d_out,float* d_in){
	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	d_out[idx]=expf(d_in[idx]);
}


int main(){
	//srand(time(0));
	float din[N];
	float dout[N];
	for (int i=0;i<N;i++){
		din[i]=5;
	}


	exponent<<<10,100>>>(dout,din);

	for (int i=0;i<N;i++){
		cout<<"expf("<<din[i]<<") "<<dout[i]<<endl;
	}

	return 0;
}
