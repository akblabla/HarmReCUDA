#include "findHighestEnergyFundamentals_d.hpp"
#include "parabularSearch.hpp"
#include "parabularSearchDefinitions.h"
#include <exception>
#include <stdio.h>
#include "cuda_runtime.h"
#include <iostream>
#include "matlabInterface.hpp"

extern "C" void squareElements_d(matrix a_d);
extern "C" void partialMatrixSummation_cuda(matrix dest_d, const matrix src_d, unsigned int rowFactor, unsigned int columnFactor);
extern "C" void findMaximum_cuda(matrix a_d);
extern "C" void elementWiseMultiplication_cuda(matrix dest_d, const matrix src_d, unsigned int rowFactor, unsigned int columnFactor);
extern "C" void findHighestEnergyFrequencies_cuda(matrix highestEnergies_d, matrix centralFreq_d, const matrix energies_d, const matrix freq_d);

void findHighestEnergyFundamentals_d(Vector_d& fundamentalFrequencies_d, const Matrix_d& amplitudes_d, const Vector_d& freq_d, int harmonicCount)
{
	if (amplitudes_d.getRows() % (2*harmonicCount) != 0) {
		throw std::exception("The harmonic energy matrix must be a factor of the amount of harmonics times 2");
		return;
	}
	/*if (src.getRows() != destination.getRows() || src.getColumns() != destination.getColumns()) {
		throw std::exception("Matrix dimension mismatch.");
		return;
	}*/
	Matrix_d harmonicEnergies_d(amplitudes_d, Matrix::matrixInitialisation::M_ASSIGN);
	squareElements_d(harmonicEnergies_d.getCMatrix());
	#ifdef _DEBUG
	printf("Harmonic Energies\n");
	harmonicEnergies_d.print(0,10, 100, 110);
	printf("\n");
	#endif // DEBUG

	Matrix_d energies_d(harmonicEnergies_d.getRows()/ (2 * harmonicCount), harmonicEnergies_d.getColumns(), Matrix::matrixInitialisation::M_ALLOCATE);
	partialMatrixSummation_cuda(energies_d.getCMatrix(), harmonicEnergies_d.getCMatrix(), (2 * (unsigned int) harmonicCount), 1);
	Matrix energies(energies_d, AMatrix::M_ASSIGN);
	matSave("D:\\Documents\\Bachelor\\Projects\\bin\\win64\\Release\\modelEnergy.mat", "modelEnergy", energies);
	#ifdef _DEBUG
	printf("Energies\n");
	energies_d.print(0,-1, 100, 110);
	printf("\n");
	#endif

	//findMaximum_cuda(energies_d.getCMatrix());
	Matrix_d highestEnergy_d(2* SEARCH_WIDTH+1, energies_d.getColumns(),AMatrix::M_ALLOCATE);
	Vector_d centralFreq_d(energies_d.getColumns(), AMatrix::M_ALLOCATE);

	findHighestEnergyFrequencies_cuda(highestEnergy_d.getCMatrix(), centralFreq_d.getCMatrix(), energies_d.getCMatrix(), freq_d.getCMatrix());
	#ifdef _DEBUG
	printf("Highest Energies\n");
	highestEnergy_d.print(0, -1, 100, 110);
	printf("\n");
	printf("Central frequencies\n");
	centralFreq_d.print(0, -1, 100, 110);
	printf("\n");
	#endif
	Matrix freq(freq_d, AMatrix::M_ASSIGN);
	double deltaFundamentalFrequency = freq.getElement(1, 0) - freq.getElement(0, 0);
	#ifdef _DEBUG
		printf("delta fundamental frequency\n");
		printf("%f", deltaFundamentalFrequency);
		printf("\n");
	#endif // DEBUG
	parabularSearch(fundamentalFrequencies_d, highestEnergy_d, centralFreq_d, deltaFundamentalFrequency);
	harmonicEnergies_d.deallocate();
	energies_d.deallocate();
}
