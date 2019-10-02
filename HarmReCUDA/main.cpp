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

#define ROWS 10000
#define COLUMNS 30000


int main() {
	printf("starting");

	cudaError_t cudaStat;
	double* matrixElements = (double*)malloc(ROWS * COLUMNS * sizeof(double));
	matrix m = { matrixElements,ROWS,COLUMNS };
	cudaEvent_t start, stop; 
	float time;
	Matrix_d d_m(ROWS, COLUMNS);
	d_m.allocateMatrixOnDevice();
	cudaEventRecord(start, 0);
	generateProjectionMatrix_cuda(d_m.getCMatrix(), 49, 51, 0, 1.0 / 31250.0, 50);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaStat = cudaMemcpy(m.elements, d_matrixElements, ROWS * COLUMNS * sizeof(*matrixElements), cudaMemcpyDeviceToHost);
	if (cudaStat != cudaSuccess) {
		printf("result download failed");
		return EXIT_FAILURE;
	}
	cudaFree(d_matrixElements);
	printf("done\n");
	cudaEventElapsedTime(&time, start, stop);
	printf("Time elapsed %f\n", time);
	for (long j = 0; j < 8; j++) {
		for (long i = 0; i < 8; i++) {
			printf("%f\t", m.elements[j * COLUMNS + i]);
		}
		printf("\n");
	}

	FILE* fptr;
	fptr = fopen("projectionMatrix.txt", "w");
	if (fptr == NULL)
	{
		printf("Error!");
		exit(1);
	}
	for (long j = 0; j < 8; j++) {
		for (long i = 0; i < COLUMNS; i++) {
			fprintf(fptr, "%f\t", m.elements[j * COLUMNS + i]);
		}
		fprintf(fptr, "\n");
	}
	fclose(fptr);
	return 1;
}
