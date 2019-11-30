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
	virtual ~AMatrix();
	virtual matrix getCMatrix() const;
	virtual int getRows() const;
	virtual int getColumns() const;
	virtual int getLeadingDimension() const;
	virtual long getElementsCount() const;
	virtual void allocate() = 0;
	virtual bool isAllocated() const;
	virtual void deallocate() = 0;
	virtual void print(int rows = 16, int columns = 8) const = 0;
	virtual void print(int rowsStart, int rowsEnd, int columnsStart, int columnsEnd) const = 0;
	virtual void transpose();
	virtual bool isTransposed() const;
protected:
	matrix _Cmatrix;
	bool _allocated = false;
};

