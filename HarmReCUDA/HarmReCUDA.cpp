#include "HarmReCUDA.h"
#include "Matrix_d.h"
#include "Vector_d.h"
#include "Vector.h"
#include <iostream>
#include <stdlib.h>
#include "GenerateProjectionMatrix_d.h"
#include <cuda_runtime.h>
#include "findHighestEnergyFundamentals_d.h"
#include "blackmanWindow_d.h"

void harmReCUDA(Matrix& data, double minimumFundamentalFrequency, double maximumFundamentalFrequency, int fundamentalFrequencyResolution, double sampleRate, Vector& harmonics)
{
	printf("starting\n");
	printf("data in\n");
	data.print();
	printf("\n");
	Matrix_d data_d(data, Matrix::M_ASSIGN);

	Vector_d harmonics_d(harmonics, Matrix::M_ASSIGN);

	
	Matrix_d projectionMatrix_d(harmonics.getRows() * fundamentalFrequencyResolution * 2, data.getRows(), Matrix::M_ALLOCATE);


	Matrix harmonicAmplitudes(projectionMatrix_d.getRows(), data.getColumns(), Matrix::M_ALLOCATE);

	Matrix_d harmonicAmplitudes_d(harmonicAmplitudes, Matrix::M_ALLOCATE);

	
	generateProjectionMatrix_d(projectionMatrix_d, minimumFundamentalFrequency, maximumFundamentalFrequency, 0, sampleRate, harmonics_d);
	blackmanWindow_d(projectionMatrix_d);
	printf("Projection Matrix\n");
	projectionMatrix_d.print();
	printf("\n");

	Matrix projectionMatrix(projectionMatrix_d, Matrix::M_ASSIGN);

	harmonicAmplitudes_d.GeneralMatrixToMatrixMultiply(projectionMatrix_d, data_d, 2.0 / data.getRows(), 0);



	Matrix_d maskedHarmonicAmplitudes_d(harmonicAmplitudes_d, Matrix::M_ALLOCATE);
	findHighestEnergyFundamentals_d(maskedHarmonicAmplitudes_d, harmonicAmplitudes_d, harmonics_d.getRows());
	printf("Amplitudes\n");
	harmonicAmplitudes_d.print(0, -1, 0, 1);
	printf("\n");
	printf("Masked Amplitudes\n");
	maskedHarmonicAmplitudes_d.print(0, -1, 0, 1);
	printf("\n");


	generateProjectionMatrix_d(projectionMatrix_d, minimumFundamentalFrequency, maximumFundamentalFrequency, 0, sampleRate, harmonics_d);

	data_d.GeneralMatrixToMatrixMultiply(projectionMatrix_d, maskedHarmonicAmplitudes_d, -1, 1.0,Matrix_d::TRANS, Matrix_d::NO_TRANS);

	printf("Done\n");

	printf("Result\n");
	data_d.print(50,5);
	printf("\n");

	harmonicAmplitudes.deallocate();
	projectionMatrix_d.deallocate();
	harmonics_d.deallocate();
	harmonicAmplitudes_d.deallocate();
	maskedHarmonicAmplitudes_d.deallocate();

}
