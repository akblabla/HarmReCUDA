#pragma once
#include "LinearAlgebraStructs.h"
#include "AMatrix.hpp"
#include "Matrix.hpp"
class Matrix;
extern class Matrix_d : public AMatrix
{
public:
	Matrix_d(int rows, int columns, matrixInitialisation initialisation = matrixInitialisation::M_NO_INIT);
	Matrix_d(const Matrix& src, matrixInitialisation initialisation = matrixInitialisation::M_NO_INIT);
	Matrix_d(const Matrix_d& src, matrixInitialisation initialisation = matrixInitialisation::M_NO_INIT);
	void allocate() override;
	void deallocate() override;
	virtual void uploadToDevice(const Matrix& src);
	virtual void copyFromDevice(const Matrix_d& src);
	virtual void GeneralMatrixToMatrixMultiply(Matrix_d& A, Matrix_d& B, double alpha, double beta);
	virtual void GeneralMatrixToMatrixMultiply(Matrix_d& A, Matrix_d& B, double alpha, double beta, matrixTranspose transposeA, matrixTranspose transposeB);
	virtual void print(int rows = 10, int columns = 5) override;
	virtual void print(int rowsStart, int rowsEnd, int columnsStart, int columnsEnd) override;
};

