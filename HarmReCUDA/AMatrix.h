#pragma once
#include"LinearAlgebraStructs.h"
class AMatrix
{
public:
	AMatrix(int rows, int columns);
	virtual matrix getCMatrix() const;
	virtual const int getRows() const;
	virtual const int getColumns() const;
	virtual const long getElementsCount() const;
protected:
	matrix _CMatrix;
};

