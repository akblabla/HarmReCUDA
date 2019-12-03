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
#include "temp.h"
#include <exception>
#define _USE_MATH_DEFINES
#include <math.h>
#include <Windows.h>

#include <chrono>  // for high_resolution_clock




int main() {


	temp(0, 0);
	const double fs = 31250;

	Matrix data = matLoad("D:\\Documents\\Bachelor\\Projects\\bin\\win64\\Release\\in.mat","data");
	Matrix harmonicsMat = matLoad("D:\\Documents\\Bachelor\\Projects\\bin\\win64\\Release\\in.mat","harmonics");
	Matrix fMinMat = matLoad("D:\\Documents\\Bachelor\\Projects\\bin\\win64\\Release\\in.mat", "fMin");
	auto fMin = fMinMat.getElement(0, 0);
	Matrix fMaxMat = matLoad("D:\\Documents\\Bachelor\\Projects\\bin\\win64\\Release\\in.mat", "fMax");
	auto fMax = fMaxMat.getElement(0, 0);
	Matrix fResMat = matLoad("D:\\Documents\\Bachelor\\Projects\\bin\\win64\\Release\\in.mat", "fRes");
	int fRes = fResMat.getElement(0, 0);
	auto start = std::chrono::high_resolution_clock::now();
	//d.print();
	Vector harmonics(harmonicsMat.getColumns());
	harmonics.allocate();
	for (int i = 0; i < harmonics.getRows(); ++i) {
		harmonics.getCMatrix().elements[i] = harmonicsMat.getElement(0,i);
	}
	try{
	harmReCUDA(data, fMin, fMax, fRes, fs, harmonics);
	}
	catch (std::exception e) {
		printf("%s", e.what());
	}
	// Record end time
	auto finish = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = finish - start;
	std::cout << "Harmonic removal execution time: " << elapsed.count() << " seconds" << std::endl;
	Matrix time(1, 1, Matrix::M_ALLOCATE);
	time.setElement(elapsed.count(), 0, 0);
	matSave("D:\\Documents\\Bachelor\\Projects\\bin\\win64\\Release\\out.mat", "data", data);
	matSave("D:\\Documents\\Bachelor\\Projects\\bin\\win64\\Release\\runtimePerformance.mat","elapsed", time);
	data.deallocate();



	std::cout << std::endl;

	return 1;
}
