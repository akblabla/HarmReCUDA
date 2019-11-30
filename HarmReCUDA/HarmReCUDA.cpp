#include "HarmReCUDA.hpp"
#include "Matrix_d.hpp"
#include "Vector_d.hpp"
#include "Vector.hpp"
#include <iostream>
#include <stdlib.h>
#include "GenerateDesignMatrix_d.hpp"
#include <cuda_runtime.h>
#include "findHighestEnergyFundamentals_d.hpp"
#include "blackmanWindow_d.hpp"
//#include "moorePenroseInversion.h"

#define GPU_MEMORY_SLACK 10000000
extern "C" void elementWiseMultiplication_cuda(matrix dest_d, const matrix src_d, unsigned int rowFactor, unsigned int columnFactor);

void harmReCUDA(Matrix& data, double minimumFundamentalFrequency, double maximumFundamentalFrequency, int fundamentalFrequencyResolution, double sampleRate, Vector& harmonics)
{

	size_t free, total;

	printf("\n");

	cudaMemGetInfo(&free, &total);

	printf("%d KB free of total %d MB\n", free / (1024*1024), total / (1024 * 1024));

	#ifdef _DEBUG
	printf("starting\n");
	printf("data in\n");
	data.print();
	printf("\n");
	#endif
	
	Matrix_d data_d(data, Matrix::M_ASSIGN);


	Vector time(data_d.getRows(), Matrix::M_ALLOCATE);
	for (int i = 0; i < time.getRows(); ++i) {
		time.setElement(i / sampleRate, i, 0);
	}
	Vector_d time_d(time, Matrix::M_ASSIGN);
	time.deallocate();
#ifdef _DEBUG
	printf("time vector\n");
	time_d.print();
	printf("\n");

#endif // DEBUG



	Vector fundamentalFrequencies(fundamentalFrequencyResolution, Matrix::M_ALLOCATE);
	const double deltaFundamentalFrequency = (maximumFundamentalFrequency - minimumFundamentalFrequency) / (double)fundamentalFrequencyResolution;
	for (int i = 0; i < fundamentalFrequencyResolution; ++i) {
		fundamentalFrequencies.setElement(minimumFundamentalFrequency + deltaFundamentalFrequency * i, i, 0);
	}
	Vector_d fundamentalFrequencies_d(fundamentalFrequencies, Matrix::M_ASSIGN);
	fundamentalFrequencies.deallocate();
#ifdef _DEBUG
	printf("fundamental frequencies vector\n");
	fundamentalFrequencies_d.print(-1,1);
	printf("\n");

#endif // DEBUG




	Vector_d harmonics_d(harmonics, Matrix::M_ASSIGN);
	Matrix_d harmonicAmplitudes_d(harmonics.getRows() * fundamentalFrequencyResolution * 2, data.getColumns(), Matrix::M_ALLOCATE);
	{
		cudaMemGetInfo(&free, &total);
		const size_t GPUMemoryForProjectionPerTestFreq = harmonics.getRows() * 2 * time_d.getElementsCount() * sizeof(double);
		const size_t GPUMemoryForProjection = free - GPU_MEMORY_SLACK;
		const size_t partitionCount = fundamentalFrequencies_d.getElementsCount() * GPUMemoryForProjectionPerTestFreq / GPUMemoryForProjection + 1;
		const size_t partitionFreqCount = fundamentalFrequencies_d.getElementsCount() / partitionCount;

		if (partitionCount > 1) {
			printf("GPU memory free %ld MB\n", GPUMemoryForProjection / 1048576);
			printf("GPU memory used per test frequencies %ld MB\n", GPUMemoryForProjectionPerTestFreq / 1048576);
			printf("Partitioning test frequencies into %ld batches of %ld\n", partitionCount, partitionFreqCount);
			printf("\n");
		}
		printf("Projecting on test frequencies");
		for (int freqsSolved = 0; freqsSolved < fundamentalFrequencyResolution; freqsSolved += partitionFreqCount) {

			Vector_d fundamentalFrequenciesToBeSolved_d(fundamentalFrequencyResolution, Matrix::M_NO_INIT);
			fundamentalFrequencies_d.getSubVector(fundamentalFrequenciesToBeSolved_d, freqsSolved, freqsSolved + partitionFreqCount - 1);
			Matrix_d designMatrix_d(data.getRows(),harmonics.getRows() * fundamentalFrequenciesToBeSolved_d.getElementsCount() * 2, Matrix::M_ALLOCATE);


			//Matrix harmonicAmplitudes(projectionMatrix_d.getRows(), data.getColumns(), Matrix::M_ALLOCATE);



			generateDesignMatrix_d(designMatrix_d, fundamentalFrequenciesToBeSolved_d, time_d, harmonics_d);
			//moorePenroseInversion_d(designMatrix_d, harmonics_d, fundamentalFrequenciesToBeSolved_d);
	#ifdef _DEBUG
			printf("Projection Matrix\n");
			designMatrix_d.print();
			printf("\n");
	#endif
			blackmanWindow_d(designMatrix_d);

			Matrix_d harmonicAmplitudesToBeSolved_d(0, 0, AMatrix::M_NO_INIT);
			harmonicAmplitudes_d.getSubMatrix(harmonicAmplitudesToBeSolved_d, freqsSolved, freqsSolved + partitionFreqCount-1, 0, -1);
			designMatrix_d.transpose();
			harmonicAmplitudes_d.GeneralMatrixToMatrixMultiply(designMatrix_d, data_d, 2.0 / time_d.getRows(), 0);
			designMatrix_d.deallocate();
			if (partitionCount > 1)
			printf(".");
		}
	}
	printf(" done!\n\n");
	Vector_d maxFundamentalFreq_d(data_d.getColumns() , Matrix::M_ALLOCATE);
	findHighestEnergyFundamentals_d(maxFundamentalFreq_d, harmonicAmplitudes_d, fundamentalFrequencies_d, harmonics_d.getRows());
	#ifdef _DEBUG
	printf("Amplitudes\n");
	harmonicAmplitudes_d.print(0, 50, 0, 5);
	printf("\n");
	printf("fundamental frequencies found\n");
	maxFundamentalFreq_d.print(0,-1, 0, -1);
	printf("\n");
	#endif
	
	
	cudaMemGetInfo(&free, &total);
	const size_t GPUMemoryForProjectionPerStack = harmonics.getRows() * 2 * maxFundamentalFreq_d.getElementsCount() * sizeof(double)
		+ harmonics.getRows() * 2* time_d.getElementsCount() * sizeof(double);
	const size_t GPUMemoryForProjection = free - GPU_MEMORY_SLACK;
	const size_t partitionCount = maxFundamentalFreq_d.getElementsCount() *GPUMemoryForProjectionPerStack / GPUMemoryForProjection + 1;
	const size_t partitionStackCount = maxFundamentalFreq_d.getElementsCount() /partitionCount;
	
	if (partitionCount>1){
		printf("GPU memory free %ld MB\n", GPUMemoryForProjection/ 1048576);
		printf("GPU memory used per stack %ld MB\n", GPUMemoryForProjectionPerStack/ 1048576);
		printf("Partitioning stacks into %ld batches of %ld\n", partitionCount, partitionStackCount);
		printf("\n");
	}
	printf("Subtracting Harmonic Model");
	for (int stacksSolved = 0; stacksSolved < maxFundamentalFreq_d.getElementsCount(); stacksSolved+= partitionStackCount){
		Vector_d maxFundamentalFreqToBeSolved_d(0, AMatrix::M_NO_INIT);
		maxFundamentalFreq_d.getSubVector(maxFundamentalFreqToBeSolved_d, stacksSolved, stacksSolved+partitionStackCount-1);
		Matrix_d newDesignMatrix_d(time_d.getElementsCount(), harmonics_d.getRows() * 2* maxFundamentalFreqToBeSolved_d.getElementsCount(), Matrix::M_ALLOCATE);
		generateDesignMatrix_d(newDesignMatrix_d, maxFundamentalFreqToBeSolved_d, time_d, harmonics_d);
	#ifdef _DEBUG
		printf("New projection matrix\n");
		newDesignMatrix_d.print();
		printf("\n");
	#endif
		blackmanWindow_d(newDesignMatrix_d);
		Matrix_d newHarmonicAmplitudes_d(harmonics_d.getRows() * 2 * maxFundamentalFreqToBeSolved_d.getElementsCount(), maxFundamentalFreqToBeSolved_d.getElementsCount(), Matrix::M_ALLOCATE);
		

		Matrix_d dataToBeSolved_d(0, 0, AMatrix::M_NO_INIT);
		data_d.getSubMatrix(dataToBeSolved_d, 0, -1, stacksSolved, stacksSolved + partitionStackCount - 1);
		newDesignMatrix_d.transpose();
		newHarmonicAmplitudes_d.GeneralMatrixToMatrixMultiply(newDesignMatrix_d, dataToBeSolved_d, 2.0 / time_d.getRows(), 0);
		Matrix ampFilter(maxFundamentalFreqToBeSolved_d.getElementsCount(), maxFundamentalFreqToBeSolved_d.getElementsCount(),AMatrix::M_ASSIGN);
		for (int i = 0; i < ampFilter.getColumns(); ++i) {
			ampFilter.setElement(1, i, i);
		}
		Matrix_d ampFilter_d(ampFilter, AMatrix::M_ASSIGN);
		elementWiseMultiplication_cuda(newHarmonicAmplitudes_d.getCMatrix(), ampFilter_d.getCMatrix(), harmonics.getRows() * 2, 1);
	
	#ifdef _DEBUG
		printf("New harmonic amplitudes\n");
		newHarmonicAmplitudes_d.print(10, 5);
		printf("\n");
	#endif
		newDesignMatrix_d.transpose();
		generateDesignMatrix_d(newDesignMatrix_d, maxFundamentalFreqToBeSolved_d, time_d, harmonics_d);
		
		dataToBeSolved_d.GeneralMatrixToMatrixMultiply(newDesignMatrix_d, newHarmonicAmplitudes_d, -1, 1.0);
		newDesignMatrix_d.deallocate();
		ampFilter_d.deallocate();
		ampFilter_d.deallocate();
		if (partitionCount > 1)
		printf(".");
	}
	printf(" done!\n\n");
	#ifdef _DEBUG


	printf("Result\n");
	data_d.print(50,5);
	printf("\n");
	#endif
	data.downloadFromDevice(data_d);
	harmonics_d.deallocate();
	harmonicAmplitudes_d.deallocate();

}