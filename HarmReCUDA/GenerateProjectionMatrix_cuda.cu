#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "GenerateProjectionMatrix_d.h"

__global__
void generateProjectionMatrix_kernel(matrix a_d, const double minFreq, const double maxFreq, const double startTime, const double deltaTime, const int harmonics)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int column_index = i%a_d.columns;
	int row_index = 2*(i/a_d.columns); //skip every second row
	double t = column_index*deltaTime+startTime;
	double freq = (minFreq+(row_index/harmonics)*(maxFreq-minFreq)/a_d.rows)*((1+row_index/2)%harmonics);
	double phase = t*freq;
	if (row_index<a_d.rows) //make sure not to write outside of matrix, incase the number of elements did not have a base of 256
	sincos(
		phase,
		&(a_d.elements[a_d.columns*(row_index+1)+column_index]),
		&(a_d.elements[a_d.columns*row_index+column_index])
	);
}
/**
*creates a projection matrix on the gpu to the given matrix on the device.
**/
extern "C" void generateProjectionMatrix_cuda(matrix a_d, const double minFreq, const double maxFreq, const double startTime, const double deltaTime, const int harmonics){
	int N = a_d.rows*a_d.columns/2; //each thread handles two elements of the matrix
	generateProjectionMatrix_kernel<<<(N+255)/256, 256>>>(a_d, minFreq, maxFreq, startTime, deltaTime, harmonics);
}