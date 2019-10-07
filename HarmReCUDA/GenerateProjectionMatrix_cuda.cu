#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "GenerateProjectionMatrix_d.h"

__global__
void generateProjectionMatrix_kernel(matrix dest_d, const double minFreq, const double maxFreq, const double startTime, const double deltaTime, const matrix harmonics_d)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int columnIndex = i/dest_d.rows;
	int rowIndex = ((2 * i) % dest_d.rows); //skip every second row
	double t = columnIndex*deltaTime+startTime;
	int fundamentalFrequencyIndex = rowIndex / (2*harmonics_d.rows);
	int fundamentalFrequencyEndIndex = (dest_d.rows-1) / (2 * harmonics_d.rows);
	int harmonicIndex = (rowIndex/2) % (harmonics_d.rows);
	double fundamentalFrequency;
	if (fundamentalFrequencyEndIndex>0){
		fundamentalFrequency = minFreq + (fundamentalFrequencyIndex) * (maxFreq - minFreq) / (fundamentalFrequencyEndIndex);
	}
	else {
		fundamentalFrequency = minFreq;
	}
	double freq = fundamentalFrequency * harmonics_d.elements[harmonicIndex];
	
	
	double phase = t*freq;
	if (columnIndex < dest_d.columns) //make sure not to write outside of matrix, incase the number of elements did not have a base of 1024
	sincos(
		phase,
		&(dest_d.elements[MATRIX_INDEX(rowIndex+1, columnIndex, dest_d.rows)]),
		&(dest_d.elements[MATRIX_INDEX(rowIndex,columnIndex,dest_d.rows)])
	);
}
/**
*creates a projection matrix on the gpu to the given matrix on the device.
**/
extern "C" void generateProjectionMatrix_cuda(matrix dest_d, const double minFreq, const double maxFreq, const double startTime, const double deltaTime, const matrix harmonics_d){
	int N = dest_d.rows* dest_d.columns/2; //each thread handles two elements of the matrix
	generateProjectionMatrix_kernel<<<(N+ 1023)/ 1024, 1024 >>>(dest_d, minFreq, maxFreq, startTime, deltaTime, harmonics_d);
}