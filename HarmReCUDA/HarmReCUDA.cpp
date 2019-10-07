#include "HarmReCUDA.h"
#include "Matrix_d.h"
#include "Vector_d.h"
#include "Vector.h"
#include <iostream>
#include <stdlib.h>
#include "GenerateProjectionMatrix_d.h"
#include <cuda_runtime.h>

#define ROWS 1000
#define COLUMNS 300

void harmReCUDA(Matrix& data)
{
	printf("starting\n");
	printf("data in\n");
	data.print();
	printf("\n");
	Matrix_d data_d(COLUMNS,1);
	data_d.allocate();
	data_d.uploadToDevice(data);

	Vector harmonics(50);
	harmonics.allocate();
	for (int i = 0; i<50; ++i){
		harmonics.getCMatrix().elements[i] = i;
	}
	printf("harmonics\n");
	harmonics.print();
	printf("\n");

	Vector_d Harmonics_d(50);
	Harmonics_d.allocate();
	Harmonics_d.uploadToDevice(harmonics);

	Matrix result(ROWS, 1);
	result.allocate();

	Matrix_d result_d(ROWS, 1);
	result_d.allocate();

	Matrix projectionMatrix(ROWS, COLUMNS);
	projectionMatrix.allocate();

	Matrix_d projectionMatrix_d(ROWS, COLUMNS);
	projectionMatrix_d.allocate();

	
	generateProjectionMatrix_d(projectionMatrix_d, 50, 50, 0, 1.0, Harmonics_d);

	result_d.GeneralMatrixToMatrixMultiply(projectionMatrix_d, data_d, 1, 0);


	result_d.downloadFromDevice(result);
	projectionMatrix_d.downloadFromDevice(projectionMatrix);



	printf("Done\n");
	projectionMatrix_d.deallocate();
	Harmonics_d.deallocate();
	result_d.deallocate();

	printf("Result\n");
	result.print();
	printf("\n");

	printf("Projection Matrix\n");
	projectionMatrix.print();
	printf("\n");
	result.deallocate();

}
