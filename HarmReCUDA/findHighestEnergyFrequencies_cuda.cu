#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "LinearAlgebraStructs.h"
#include "parabularSearchDefinitions.h"

__global__
void findHighestEnergyFrequencies_kernel(matrix highestEnergies_d, matrix centralFreq_d, const matrix energies_d, const matrix freq_d)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int highestEnergiesColumnIndex = i;
	int energiesColumnIndex = highestEnergiesColumnIndex;

	double maxEnergy = 0;
	int maxIndex = 0;
	if (highestEnergiesColumnIndex < highestEnergies_d.columns) { //don't write outside matrix
		for (int energiesRowIndex = 0; energiesRowIndex < energies_d.rows; energiesRowIndex++) {
			double energy = energies_d.elements[MATRIX_INDEX(energiesRowIndex, energiesColumnIndex, energies_d.ld)];
			if (energy > maxEnergy) {
				maxEnergy = energy;
				maxIndex = energiesRowIndex;
			}
		}
		int centralFreqColumnIndex = highestEnergiesColumnIndex;
		int centralFreqIndex = maxIndex;
		if (centralFreqIndex < SEARCH_WIDTH) centralFreqIndex = SEARCH_WIDTH;
		if (centralFreqIndex > freq_d.rows-SEARCH_WIDTH-1) centralFreqIndex = freq_d.rows - SEARCH_WIDTH - 1;

		centralFreq_d.elements[centralFreqColumnIndex] = freq_d.elements[centralFreqIndex];

		for (int highestEnergiesRowIndex = 0; highestEnergiesRowIndex < highestEnergies_d.rows; highestEnergiesRowIndex++) {
			highestEnergies_d.elements[MATRIX_INDEX(highestEnergiesRowIndex, highestEnergiesColumnIndex, highestEnergies_d.ld)] = energies_d.elements[MATRIX_INDEX(highestEnergiesRowIndex+centralFreqIndex-SEARCH_WIDTH, highestEnergiesColumnIndex, energies_d.ld)];
		}
	}
}


extern "C" void findHighestEnergyFrequencies_cuda(matrix highestEnergies_d, matrix centralFreq_d, const matrix energies_d, const matrix freq_d) {
	int N = energies_d.columns;
	findHighestEnergyFrequencies_kernel << < (N + 255) / 256, 256 >> > (highestEnergies_d, centralFreq_d, energies_d, freq_d);
}