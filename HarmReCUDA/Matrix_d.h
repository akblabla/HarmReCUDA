#pragma once
#include "LinearAlgebraStructs.h"
#include "Matrix.h"
class Matrix_d : public Matrix
{
public:
	Matrix_d(int rows, int columns, matrixInitialisation::matrixInitialisation initialisation = matrixInitialisation::no_init);
	Matrix_d(const Matrix& src, matrixInitialisation::matrixInitialisation initialisation = matrixInitialisation::no_init);
	Matrix_d(const Matrix_d& src, matrixInitialisation::matrixInitialisation initialisation = matrixInitialisation::no_init);
	void allocate() override;
	void deallocate() override;
	virtual void uploadToDevice(const Matrix& src);
	virtual void copyFromDevice(const Matrix_d& src);
	virtual void downloadFromDevice(Matrix& dst);
	virtual void GeneralMatrixToMatrixMultiply(Matrix_d& A, Matrix_d& B, double alpha, double beta);

};

