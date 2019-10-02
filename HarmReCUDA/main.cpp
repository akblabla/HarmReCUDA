/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/* Example of integrating CUDA functions into an existing
 * application / framework.
 * CPP code representing the existing application / framework.
 * Compiled with default CPP compiler.
 */

// includes, system
#include <iostream>
#include <stdlib.h>

// Required to include CUDA vector types
#include <cuda_runtime.h>
#include <vector_types.h>
//#include <helper_cuda.h>
#include "LinearAlgebraStructs.h"

#include "HarmReCUDA.h"

int main() {
	NRMat<double> data(5,10, 20);
	NRVec<double> harmonics(5, 20);
	NRVec<double> frequencies(5, 20);
	//data = harmReCUDA(data, harmonics, frequencies);
	return 1;
}
