#pragma once
#include "LinearAlgebraStructs.h"
#include "AMatrix.hpp"
#include "Matrix_d.hpp"

class Matrix_d;
extern class Matrix : public AMatrix
{
public:
	Matrix(int rows, int columns, matrixInitialisation initialisation = matrixInitialisation::M_NO_INIT);
	Matrix(const Matrix_d& src, matrixInitialisation initialisation = matrixInitialisation::M_NO_INIT);
	virtual void allocate();
	virtual void deallocate();
	virtual double getElement(int row, int column) const;
	virtual void downloadFromDevice(const Matrix_d& src);
	virtual void setElement(double value, int row, int column);
	virtual void getSubMatrix(Matrix& dest, int rowsStart, int rowsEnd, int columnsStart, int columnsEnd) const;
	void print(int rows = 10, int columns = 5) const override;
	void print(int rowsStart, int rowsEnd, int columnsStart, int columnsEnd) const override;
	virtual void printToFile(const char* filename, int rows = 10, int columns = 5);
	virtual ~Matrix();
};

