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
#include <iostream>

#include "matlabInterface.hpp"
#include "HarmReCUDA.hpp"
#include "Matrix.hpp"
#include "Vector.hpp"
#include <exception>
#define _USE_MATH_DEFINES
#include <math.h>
#include <Windows.h>

#include <chrono>  // for high_resolution_clock




int main() {
	const double fs = 31250;

	Matrix d = matLoad("D:\\Documents\\Bachelor\\Projects\\bin\\win64\\Release\\in.mat");
	auto start = std::chrono::high_resolution_clock::now();
	//d.print();
	Vector harmonics(50);
	harmonics.allocate();
	for (int i = 0; i < harmonics.getRows(); ++i) {
		harmonics.getCMatrix().elements[i] = i + 1;
	}

	harmReCUDA(d, 49.9 * (2 * M_PI), 50.1 * (2 * M_PI), 110, fs, harmonics);

	// Record end time
	auto finish = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = finish - start;
	std::cout << "Harmonic removal execution time: " << elapsed.count() << " seconds" << std::endl;

	matSave("D:\\Documents\\Bachelor\\Projects\\bin\\win64\\Release\\out.mat",d);
	d.deallocate();



	std::cout << std::endl;

	return 1;
}
