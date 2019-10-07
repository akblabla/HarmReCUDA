#pragma once
#include "LinearAlgebraStructs.h"
namespace matrixInitialisation {
	enum matrixInitialisation{ no_init, allocate, assign };
}
class Matrix
{
public:
	Matrix(int rows, int columns, matrixInitialisation::matrixInitialisation initialisation = matrixInitialisation::no_init);
	virtual matrix getCMatrix() const;
	virtual int getRows() const;
	virtual int getColumns() const;
	virtual long getElementsCount() const;
	virtual void allocate();
	virtual void deallocate();
	virtual double getElement(int row, int column);
	virtual void setElement(double value, int row, int column);
	virtual void print(int rows = 16, int columns = 8);
	virtual ~Matrix();
protected:
	matrix _Cmatrix;
	bool _allocated = false;
};

