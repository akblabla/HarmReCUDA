#pragma once
#include "LinearAlgebraStructs.h"
class Matrix
{
public:
	Matrix(int rows, int columns);
	virtual matrix getCMatrix() const;
	virtual const int getRows() const;
	virtual const int getColumns() const;
	virtual const long getElementsCount() const;
	virtual ~Matrix();
protected:
	matrix _Cmatrix;
};

