#pragma once
#include "LinearAlgebraStructs.h"
#include "Matrix.h"
class Matrix_d : Matrix
{
public:
	Matrix_d(int rows, int columns);
	virtual const int getRows() const;
	virtual const int getColumns() const;
	virtual const long getElementsCount() const;
	virtual void allocateMatrix();
	virtual void deallocateMatrix();
	virtual void uploadMatrixToDevice(Matrix& src) const;
	virtual void downloadMatrixFromDevice(Matrix& dst) const;
private:
	matrix _Cmatrix;
	bool _allocatedToDevice = false;

};

