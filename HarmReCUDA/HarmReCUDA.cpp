#include "HarmReCUDA.hpp"
#include "Matrix_d.hpp"
#include "Vector_d.hpp"
#include "Vector.hpp"
#include <iostream>
#include <stdlib.h>
#include "GenerateProjectionMatrix_d.hpp"
#include <cuda_runtime.h>
#include "findHighestEnergyFundamentals_d.hpp"
#include "blackmanWindow_d.hpp"

void harmReCUDA(Matrix& data, double minimumFundamentalFrequency, double maximumFundamentalFrequency, int fundamentalFrequencyResolution, double sampleRate, Vector& harmonics)
{

	#ifdef DEBUG
	printf("starting\n");
	printf("data in\n");
	data.print();
	printf("\n");
	#endif
	
	Matrix_d data_d(data, Matrix::M_ASSIGN);

	Vector_d harmonics_d(harmonics, Matrix::M_ASSIGN);

	
	Matrix_d projectionMatrix_d(harmonics.getRows() * fundamentalFrequencyResolution * 2, data.getRows(), Matrix::M_ALLOCATE);


	Matrix harmonicAmplitudes(projectionMatrix_d.getRows(), data.getColumns(), Matrix::M_ALLOCATE);

	Matrix_d harmonicAmplitudes_d(harmonicAmplitudes, Matrix::M_ALLOCATE);

	
	generateProjectionMatrix_d(projectionMatrix_d, minimumFundamentalFrequency, maximumFundamentalFrequency, 0, sampleRate, harmonics_d);
	#ifdef DEBUG
	printf("Projection Matrix\n");
	projectionMatrix_d.print();
	printf("\n");
	#endif
	blackmanWindow_d(projectionMatrix_d);

	Matrix projectionMatrix(projectionMatrix_d, Matrix::M_ASSIGN);

	harmonicAmplitudes_d.GeneralMatrixToMatrixMultiply(projectionMatrix_d, data_d, 2.0 / data.getRows(), 0);

	Matrix_d maskedHarmonicAmplitudes_d(harmonicAmplitudes_d, Matrix::M_ALLOCATE);
	findHighestEnergyFundamentals_d(maskedHarmonicAmplitudes_d, harmonicAmplitudes_d, harmonics_d.getRows());
	#ifdef DEBUG
	printf("Amplitudes\n");
	harmonicAmplitudes_d.print(0, 50, 0, 5);
	printf("\n");
	printf("Masked Amplitudes\n");
	maskedHarmonicAmplitudes_d.print(0, 50, 0, 5);
	printf("\n");
	#endif

	generateProjectionMatrix_d(projectionMatrix_d, minimumFundamentalFrequency, maximumFundamentalFrequency, 0, sampleRate, harmonics_d);

	data_d.GeneralMatrixToMatrixMultiply(projectionMatrix_d, maskedHarmonicAmplitudes_d, -1, 1.0,Matrix_d::TRANS, Matrix_d::NO_TRANS);
	#ifdef DEBUG

	printf("Done\n");

	printf("Result\n");
	data_d.print(50,5);
	printf("\n");
	#endif
	data.downloadFromDevice(data_d);
	harmonicAmplitudes.deallocate();
	projectionMatrix_d.deallocate();
	harmonics_d.deallocate();
	harmonicAmplitudes_d.deallocate();
	maskedHarmonicAmplitudes_d.deallocate();

}
