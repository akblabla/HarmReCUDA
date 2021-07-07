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
#include "moorePenroseInversion.h"
#include "approximatePseudoInversion.h"
#include "cublas_v2.h"

#define GPU_MEMORY_SLACK 10000000
#define STREAM_COUNT 10
extern "C" void elementWiseMultiplication_cuda(matrix dest_d, const matrix src_d, unsigned int rowFactor, unsigned int columnFactor);

void harmReCUDA(Matrix& data, double minimumFundamentalFrequency, double maximumFundamentalFrequency, int fundamentalFrequencyResolution, double sampleRate, Vector& harmonics)
{

	cublasHandle_t handle;
	cublasCreate(&handle);
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
		/*cudaStream_t streams[2];
		cudaStreamCreate(&(streams[0]));
		cudaStreamCreate(&(streams[1]));
		*/
		cudaMemGetInfo(&free, &total);
		const size_t GPUMemoryForProjectionPerTestFreq = (2*harmonics.getRows() * 2 * harmonics.getRows() * 2 + harmonics.getRows() * 2 * time_d.getElementsCount()) * sizeof(double);
		const size_t GPUMemoryForProjection = free - GPU_MEMORY_SLACK;
		const size_t partitionCount = (fundamentalFrequencies_d.getElementsCount() * GPUMemoryForProjectionPerTestFreq) / GPUMemoryForProjection + 1;
		const size_t partitionFreqCount = fundamentalFrequencies_d.getElementsCount() / partitionCount;

		if (partitionCount > 1) {
			printf("GPU memory free %ld MB\n", GPUMemoryForProjection / 1048576);
			printf("GPU memory used per test frequencies %ld MB\n", GPUMemoryForProjectionPerTestFreq / 1048576);
			printf("Partitioning test frequencies into %ld batches of %ld\n", partitionCount, partitionFreqCount);
			printf("\n");
		}
		printf("Projecting on test frequencies");
		Matrix_d predesignMatrix_d(data.getRows(), harmonics.getRows() * partitionFreqCount * 2, Matrix::M_ALLOCATE);
		for (int freqsSolved = 0; freqsSolved < fundamentalFrequencyResolution; freqsSolved += partitionFreqCount) {

			Vector_d fundamentalFrequenciesToBeSolved_d(fundamentalFrequencyResolution, Matrix::M_NO_INIT);
			fundamentalFrequencies_d.getSubVector(fundamentalFrequenciesToBeSolved_d, freqsSolved, freqsSolved + partitionFreqCount - 1);
			Matrix_d designMatrix_d(data.getRows(),harmonics.getRows() * fundamentalFrequenciesToBeSolved_d.getElementsCount() * 2, Matrix::M_NO_INIT);
			if (fundamentalFrequencyResolution - freqsSolved> partitionFreqCount)
				predesignMatrix_d.getSubMatrix(designMatrix_d, 0, -1, 0, partitionFreqCount* harmonics.getRows() * 2);
			else
				predesignMatrix_d.getSubMatrix(designMatrix_d, 0, -1, 0, (fundamentalFrequencyResolution - freqsSolved) * harmonics.getRows() * 2);


			generateDesignMatrix_d(designMatrix_d, fundamentalFrequenciesToBeSolved_d, time_d, harmonics_d);
			//moorePenroseInversion_d(designMatrix_d, harmonics_d, fundamentalFrequenciesToBeSolved_d);
			approximatePseudoInversion(designMatrix_d);

			Matrix_d harmonicAmplitudesToBeSolved_d(0, 0, AMatrix::M_NO_INIT);
			harmonicAmplitudes_d.getSubMatrix(harmonicAmplitudesToBeSolved_d, 2* harmonics.getRows() * freqsSolved, 2* harmonics.getRows() * (freqsSolved + partitionFreqCount-1), 0, -1);
			//designMatrix_d.transpose();
			harmonicAmplitudesToBeSolved_d.GeneralMatrixToMatrixMultiply(handle, designMatrix_d, data_d, 1, 0);
			if (partitionCount > 1)
			printf(".");
		}
		predesignMatrix_d.deallocate();
	}
	printf(" done!\n\n");
	Vector_d maxFundamentalFreq_d(data_d.getColumns() , Matrix::M_ALLOCATE);
	findHighestEnergyFundamentals_d(maxFundamentalFreq_d, harmonicAmplitudes_d, fundamentalFrequencies_d, harmonics_d.getRows());
	#ifdef _DEBUG
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
	cudaStream_t streams[STREAM_COUNT] = {};
	for (int i = 0; i < STREAM_COUNT; ++i) {
		cudaStreamCreate(&(streams[i]));
	}
	size_t streamID = 0;


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
		Matrix_d designMatrix_d(time_d.getElementsCount(), harmonics_d.getRows() * 2* maxFundamentalFreqToBeSolved_d.getElementsCount(), Matrix::M_ALLOCATE);
		generateDesignMatrix_d(designMatrix_d, maxFundamentalFreqToBeSolved_d, time_d, harmonics_d);
	#ifdef _DEBUG
		printf("New projection matrix\n");
		designMatrix_d.print();
		printf("\n");
	#endif
		Matrix_d newHarmonicAmplitudes_d(harmonics_d.getRows() * 2, maxFundamentalFreqToBeSolved_d.getElementsCount(), Matrix::M_ALLOCATE);
		

		Matrix_d dataToBeSolved_d(0, 0, AMatrix::M_NO_INIT);
		data_d.getSubMatrix(dataToBeSolved_d, 0, -1, stacksSolved, stacksSolved + partitionStackCount - 1);

		for (int i = 0; i < dataToBeSolved_d.getColumns(); ++i) {
			Matrix_d dataToBeSolvedStack_d(0, 0, AMatrix::M_NO_INIT);
			dataToBeSolved_d.getSubMatrix(dataToBeSolvedStack_d, 0, -1, i, i);
			Matrix_d designMatrixToBeUsed_d(0, 0, AMatrix::M_NO_INIT);
			designMatrix_d.getSubMatrix(designMatrixToBeUsed_d, 0, -1, i*2* harmonics_d.getRows(), (i + 1)*2* harmonics_d.getRows()-1);

			Matrix_d newHarmonicAmplitudesStack_d(0, 0, AMatrix::M_NO_INIT);
			newHarmonicAmplitudes_d.getSubMatrix(newHarmonicAmplitudesStack_d, 0, -1, i, i);

			approximatePseudoInversion(designMatrixToBeUsed_d);
			//printf("Transposed Design Matrix\n");
			//designMatrixToBeUsed_d.print();
			//printf("\n");

			cublasSetStream(handle, streams[i % STREAM_COUNT]);
			newHarmonicAmplitudesStack_d.GeneralMatrixToMatrixMultiply(handle,designMatrixToBeUsed_d, dataToBeSolvedStack_d, 1, 0);
			//printf("Harmonics Vector\n");
			//newHarmonicAmplitudesStack_d.print(0,-1,0,-1);
			//printf("\n");
			approximatePseudoInversion(designMatrixToBeUsed_d);

			//printf("Data Vector\n");
			//dataToBeSolvedStack_d.print(0,20,0,1);
			//printf("\n");

			dataToBeSolvedStack_d.GeneralMatrixToMatrixMultiply(handle,designMatrixToBeUsed_d, newHarmonicAmplitudesStack_d, -designMatrixToBeUsed_d.getRows()/2, 1);
			//dataToBeSolvedStack_d.print(0, 20, 0, 1);
			//printf("\n");
			cublasSetStream(handle, NULL);
		}
	
	#ifdef _DEBUG
		printf("New harmonic amplitudes\n");
		newHarmonicAmplitudes_d.print(10, 5);
		printf("\n");
	#endif
		designMatrix_d.deallocate();
		newHarmonicAmplitudes_d.deallocate();
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