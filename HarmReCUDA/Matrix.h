#pragma once
#include "LinearAlgebraStructs.h"
class Matrix
{
public:
	Matrix(int rows, int columns);
	virtual matrix getCMatrix() const;
	virtual int getRows() const;
	virtual int getColumns() const;
	virtual long getElementsCount() const;
	virtual void allocate();
	virtual void deallocate();
	virtual double getElement(int row, int column);
	virtual void setElement(double value, int row, int column);
	virtual void print(int rows = 20, int columns = 10);
	virtual ~Matrix();
protected:
	matrix _Cmatrix;
	bool _allocated = false;
};

