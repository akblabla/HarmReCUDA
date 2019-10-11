#pragma once
#include "LinearAlgebraStructs.h"
#include "AMatrix.h"
#include "Matrix_d.h"
class Matrix_d;
class Matrix : public AMatrix
{
public:
	Matrix(int rows, int columns, matrixInitialisation initialisation = matrixInitialisation::M_NO_INIT);
	Matrix(const Matrix_d& src, matrixInitialisation initialisation = matrixInitialisation::M_NO_INIT);
	virtual void allocate();
	virtual void deallocate();
	virtual double getElement(int row, int column);
	virtual void downloadFromDevice(const Matrix_d& src);
	virtual void setElement(double value, int row, int column);
	virtual void print(int rows = 10, int columns = 5) override;
	virtual void print(int rowsStart, int rowsEnd, int columnsStart, int columnsEnd) override;
	virtual void printToFile(const char* filename, int rows = 10, int columns = 5);
	virtual ~Matrix();
};

