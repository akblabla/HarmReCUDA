#include "HarmReCUDA.h"
#include "Matrix_d.h"
#include "Vector_d.h"
#include "Vector.h"
#include <iostream>
#include <stdlib.h>
#include "GenerateProjectionMatrix_d.h"
#include <cuda_runtime.h>
#include "findHighestEnergyFundamentals_d.h"


void harmReCUDA(Matrix& data, double minimumFundamentalFrequency, double maximumFundamentalFrequency, int fundamentalFrequencyResolution, double sampleRate, Vector& harmonics)
{
	printf("starting\n");
	printf("data in\n");
	data.print();
	printf("\n");
	Matrix_d data_d(data,matrixInitialisation::assign);
	
	printf("harmonics\n");
	harmonics.print();
	printf("\n");

	Vector_d harmonics_d(harmonics, matrixInitialisation::assign);

	Matrix projectionMatrix(harmonics.getRows()* fundamentalFrequencyResolution*2, data.getRows(), matrixInitialisation::allocate);

	Matrix_d projectionMatrix_d(projectionMatrix, matrixInitialisation::allocate);


	Matrix harmonicAmplitudes(projectionMatrix.getRows(), data.getColumns(), matrixInitialisation::allocate);

	Matrix_d harmonicAmplitudes_d(harmonicAmplitudes,matrixInitialisation::allocate);

	
	generateProjectionMatrix_d(projectionMatrix_d, minimumFundamentalFrequency, maximumFundamentalFrequency, 0, 1.0/ sampleRate, harmonics_d);

	harmonicAmplitudes_d.GeneralMatrixToMatrixMultiply(projectionMatrix_d, data_d, 1, 0);

	Matrix_d energyMask(harmonicAmplitudes_d, matrixInitialisation::allocate);

	findHighestEnergyFundamentals_d(harmonicAmplitudes_d, energyMask, harmonics_d.getRows());

	harmonicAmplitudes_d.downloadFromDevice(harmonicAmplitudes);
	projectionMatrix_d.downloadFromDevice(projectionMatrix);



	printf("Done\n");
	projectionMatrix_d.deallocate();
	harmonics_d.deallocate();
	harmonicAmplitudes_d.deallocate();

	printf("Result\n");
	harmonicAmplitudes.print();
	printf("\n");

	printf("Projection Matrix\n");
	projectionMatrix.print();
	printf("\n");
	harmonicAmplitudes.deallocate();

}
