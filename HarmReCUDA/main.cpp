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
#include "Matrix_d.h"

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
extern "C" void generateProjectionMatrix_cuda(matrix a_d, const double minFreq, const double maxFreq, const double startTime, const double deltaTime, const int harmonics);

#define ROWS 8192
#define COLUMNS 32768


int main() {
	printf("starting");

	Matrix m(ROWS,COLUMNS);
	m.allocateMatrix();
	cudaEvent_t start, stop; 
	float time;
	Matrix_d d_m(ROWS, COLUMNS);
	d_m.allocateMatrix();
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	generateProjectionMatrix_cuda(d_m.getCMatrix(), 49, 51, 0, 1.0 / 31250.0, 50);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	d_m.downloadMatrixFromDevice(m);
	//cudaFree(d_matrixElements);
	printf("done\n");
	cudaEventElapsedTime(&time, start, stop);
	printf("Time elapsed %f\n", time);

	d_m.deallocateMatrix();

	for (long j = 0; j < 8; j++) {
		for (long i = 0; i < 8; i++) {
			printf("%1.3f\t", m.getCMatrix().elements[j * COLUMNS + i]);
		}
		printf("\n");
	}
	m.deallocateMatrix();

	return 1;
}
