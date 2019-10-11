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
#include <stdio.h>

#include "HarmReCUDA.h"
#include "Matrix.h"
#include "Vector.h"
#include <exception>
#define _USE_MATH_DEFINES
#include <math.h>

int main() {
	const double fs = 31250;
	const double f = 2050;
	const double phase = 0;// 3.14159265359 / 4.0;
	Matrix d(fs, 1);
	d.allocate();
	for (int i = 0; i < d.getRows(); ++i) {
		for (int j = 0; j < d.getColumns(); ++j) {
			d.setElement(cos(f*2* M_PI *i/ fs *(j+1)+ phase),i, j);
		}
	}

	Vector harmonics(2);
	harmonics.allocate();
	for (int i = 0; i < harmonics.getRows(); ++i) {
		harmonics.getCMatrix().elements[i] = i + 1;
	}

	harmReCUDA(d, 2049.9, 2050.1, 110, fs, harmonics);
	
	d.deallocate();
	return 1;
}
