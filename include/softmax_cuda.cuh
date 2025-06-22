#pragma once
#include <iostream>
#include <random>
#include <ctime>
#include <thread>
#include <float.h>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <functional>
#include <cmath>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <cuda.h>

__global__ void softmax_fused(float* resd, const float* xd, const int M,const int N);

