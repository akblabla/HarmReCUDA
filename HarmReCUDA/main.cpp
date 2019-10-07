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
#include "Matrix.h"
#include "Vector.h"
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
	double fs = 30000;
	Matrix d(30000, 100);
	d.allocate();
	for (int i = 0; i < d.getRows(); ++i) {
		for (int j = 0; j < d.getColumns(); ++j) {
			d.setElement(cos(2*3.14*i/ fs *(j+1)),i, j);
		}
	}

	Vector harmonics(2);
	harmonics.allocate();
	for (int i = 0; i < 2; ++i) {
		harmonics.getCMatrix().elements[i] = i + 1;
	}

	harmReCUDA(d,1,1,100, fs,harmonics);
	d.deallocate();
	return 1;
}
