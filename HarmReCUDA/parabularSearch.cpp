#include "parabularSearch.hpp"

extern "C" void findHighestEnergyFrequencies_cuda(matrix highestEnergies_d, matrix centralFreq_d, const matrix energies_d, const matrix freq_d);
extern "C" void elementWiseDivision_cuda(matrix dest_d, const matrix src_d, unsigned int rowFactor, unsigned int columnFactor);

void parabularSearch(Matrix_d& freqOut_d, const Matrix_d& energies_d, const Vector_d& freq_d){
	 
	const Vector_d centralFreq_d(energies_d.getColumns(), AMatrix::M_ALLOCATE);

	Matrix inverseParabularDesign(5, 2, AMatrix::M_ALLOCATE);
	const double inverseParabularDesignDefinition[10] = {
		-0.200,	-0.100,	0,	0.1000,	0.2000,
		0.1176,	0.0294,	0,	0.0294,	0.1176
	}; //precalculated in matlab

	for (int i = 0; i < inverseParabularDesign.getColumns(); i++) {
		inverseParabularDesign.getCMatrix().elements[i] = inverseParabularDesignDefinition[i];
	}

	const Matrix_d inverseParabularDesign_d(inverseParabularDesign, AMatrix::M_ASSIGN);

	Matrix_d parameters_d(2, energies_d.getColumns(), AMatrix::M_ALLOCATE);
	parameters_d.GeneralMatrixToMatrixMultiply(inverseParabularDesign_d, energies_d, 1.0, 0.0);
}
