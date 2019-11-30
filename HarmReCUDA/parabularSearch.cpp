#include "parabularSearch.hpp"
#include <stdio.h>
#include <iostream>
#include "parabularSearchDefinitions.h"

extern "C" void findHighestEnergyFrequencies_cuda(matrix highestEnergies_d, matrix centralFreq_d, const matrix energies_d, const matrix freq_d);
extern "C" void elementWiseMultiplication_cuda(matrix dest_d, const matrix src_d, unsigned int rowFactor, unsigned int columnFactor);

extern "C" void elementWiseDivision_cuda(matrix dest_d, const matrix src_d, unsigned int rowFactor, unsigned int columnFactor);

void parabularSearch(Matrix_d& freqOut_d, const Matrix_d& energies_d, const Vector_d& freq_d, double deltaFundamentalFrequency){
	 
	const Vector_d centralFreq_d(energies_d.getColumns(), AMatrix::M_ALLOCATE);

	Matrix inverseParabularDesign(2, (SEARCH_WIDTH * 2 + 1), AMatrix::M_ALLOCATE);
	const double inverseParabularDesignDefinition[(2* SEARCH_WIDTH+1)*2] = {
		 -0.0667, - 0.0500, - 0.0333, - 0.0167,    0.0000,    0.0167,    0.0333,    0.0500,    0.0667,
	0.0303,    0.0076, - 0.0087, - 0.0184, - 0.0216, - 0.0184, - 0.0087,    0.0076,    0.0303,
	}; //precalculated in matlab

	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < 2* SEARCH_WIDTH+1; j++) {
			inverseParabularDesign.setElement(inverseParabularDesignDefinition[i * (SEARCH_WIDTH*2+1) + j], i, j);
		}
	}

	Matrix_d inverseParabularDesign_d(inverseParabularDesign, AMatrix::M_ASSIGN);

	Matrix_d parameters_d(2, energies_d.getColumns(), AMatrix::M_ALLOCATE);
#ifdef _DEBUG
	printf("inverse Parabular Design\n");
	inverseParabularDesign_d.print(-1, -1);
	printf("\n X\n");
	printf("maximum energies\n");
	energies_d.print(-1, 10);
	printf("\n");
#endif // DEBUG

	parameters_d.GeneralMatrixToMatrixMultiply(inverseParabularDesign_d, energies_d, 1.0, 0.0);

#ifdef _DEBUG
	printf("frequency fit parameters\n");
	parameters_d.print(-1, 10);
	printf("\n");
#endif // DEBUG
	Matrix_d slope_d(0, 0);
	parameters_d.getSubMatrix(slope_d,0, 0, 0, -1);
	Matrix_d accel_d(0, 0);
	parameters_d.getSubMatrix(accel_d,1, 1, 0, -1);

#ifdef _DEBUG
	printf("slope\n");
	slope_d.print(0,-1, 100, 110);
	printf("\n");
	printf("accel\n");
	accel_d.print(0, -1, 100, 110);
	printf("\n");
#endif // DEBUG
	elementWiseDivision_cuda(slope_d.getCMatrix(), accel_d.getCMatrix(), 1, 1);
#ifdef _DEBUG
	printf("slope/accel\n");
	slope_d.print(0, -1, 100, 110);
	printf("\n");
#endif // DEBUG
	freqOut_d.GeneralMatrixToMatrixAddition(freq_d, slope_d, 1.0, -0.5* deltaFundamentalFrequency,Matrix_d::NO_TRANS, Matrix_d::TRANS);

}
