#pragma once
#include "LinearAlgebraStructs.h"
#include "Matrix.h"
class Matrix_d : public Matrix
{
public:
	Matrix_d(int rows, int columns);
	void allocate() override;
	void deallocate() override;
	virtual void uploadToDevice(Matrix& src);
	virtual void downloadFromDevice(Matrix& dst);
	virtual void GeneralMatrixToMatrixMultiply(Matrix_d& A, Matrix_d& B, double alpha, double beta);

};

