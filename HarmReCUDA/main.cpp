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
#include <cuda_profiler_api.h>
#include <chrono>  // for high_resolution_clock
#include <conio.h>




int main(int argc, char* argv[]) {

	std::string path;
	if (argc == 1) {
		path.assign(".\\");
	}
	else if (argc == 2){
		path.assign((argv[1]));
		path.append("\\");
	}
	else
	{
		std::cout <<"Too many arguments" << std::endl;
		return -1;
	}
	Matrix data = matLoad(path+"in.mat","data");
	Matrix harmonicsMat = matLoad(path + "in.mat","harmonics");
	Matrix fMinMat = matLoad(path + "in.mat", "fMin");
	auto fMin = fMinMat.getElement(0, 0);
	Matrix fMaxMat = matLoad(path + "in.mat", "fMax");
	auto fMax = fMaxMat.getElement(0, 0);
	Matrix fResMat = matLoad(path + "in.mat", "fRes");
	int fRes = fResMat.getElement(0, 0);
	Matrix fsMat = matLoad(path + "in.mat", "fs");
	double fs = fsMat.getElement(0, 0);
	auto start = std::chrono::high_resolution_clock::now();
	//d.print();
	cudaProfilerStart();
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
	cudaProfilerStop();
	// Record end time
	auto finish = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = finish - start;
	std::cout << "Harmonic removal execution time: " << elapsed.count() << " seconds" << std::endl;
	Matrix time(1, 1, Matrix::M_ALLOCATE);
	time.setElement(elapsed.count(), 0, 0);
	matSave(path + "out.mat", "data", data);
	matSave(path + "runtimePerformance.mat","elapsed", time);
	data.deallocate();



	std::cout << std::endl;
	getch();
	return 0;
}
