#include<iostream>
#include<ctime>
#include<cstdlib>
#include<cuda_runtime.h>
#include<cmath>

#define N 1000

using namespace std;
__global__ void exponent(float* d_out,float* d_in){
	int idx=threadIdx.x;
	if (idx >= N) return;
	d_out[idx]=exp2f(d_in[idx]);
}


int main(){
	srand(time(0));
	float* din=new float[N];
	float* dout=new float[N];
	for (int i=0;i<N;i++){
		din[i]=rand()%5;
	}


	exponent<<<1,100>>>(dout,din);
	
	float cum_abs_err;
	float max_abs=0;
	for (int i=0;i<N;i++){
		cum_abs_err+=fabs(dout[i]-expf(din[i]));
		max_abs=fmax(max_abs,fabs(dout[i]-expf(din[i])));			
	}

	cout<<"cumm_abs_error: "<<cum_abs_err<<endl;
	cout<<"max_abs_err: "<<max_abs<<endl;
	
	delete[] din;
	delete[] dout;
	return 0;
}
