#pragma once
#include "LinearAlgebraStructs.h"
class AMatrix
{
public:
	enum matrixTranspose
	{
		NO_TRANS,
		TRANS
	};
	enum matrixInitialisation { M_NO_INIT, M_ALLOCATE, M_ASSIGN };
	AMatrix(int rows, int columns);
	virtual matrix getCMatrix() const;
	virtual int getRows() const;
	virtual int getColumns() const;
	virtual long getElementsCount() const;
	virtual void allocate() = 0;
	virtual void deallocate() = 0;
	virtual void print(int rows = 16, int columns = 8) = 0;
	virtual void print(int rowsStart, int rowsEnd, int columnsStart, int columnsEnd) = 0;
protected:
	matrix _Cmatrix;
	bool _allocated = false;
};

