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
#include "HarmReCUDA.h"
#include "Matrix_d.h"
#include <math.h>


int main() {
	/*printf("starting\n");

	Matrix m(ROWS,COLUMNS);
	m.allocateMatrix();
	for (int i = 0; i < ROWS * COLUMNS; i++) {
		m.getCMatrix().elements[i] = 0;
	}
	float time;
	Matrix_d d_m(ROWS, COLUMNS);
	Vector_d harmonics_d(50);
	d_m.allocateMatrix();
	harmonics_d.allocateMatrix();
	//d_m.uploadMatrixToDevice(m);
	generateProjectionMatrix_d(d_m, 49, 51, 0, 1.0 / 31250.0, harmonics_d);

	d_m.downloadMatrixFromDevice(m);
	//cudaFree(d_matrixElements);
	printf("done\n");
	d_m.deallocateMatrix();
	harmonics_d.deallocateMatrix();

	for (long j = 0; j < 8; j++) {
		for (long i = 0; i < 8; i++) {
			printf("%1.3f\t", m.getCMatrix().elements[j * COLUMNS + i]);
		}
		printf("\n");
	}
	m.deallocateMatrix();
	*/
	Matrix d(300, 1);
	d.allocate();
	for (int i = 0; i < 300; ++i) {
		for (int j = 0; j < 1; ++j) {
			d.setElement(5,i, j);
		}
	}
	harmReCUDA(d);
	d.deallocate();
	return 1;
}
