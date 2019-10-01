#pragma once
#include "LinearAlgebraStructs.h"
#include "Matrix.h"
class Matrix_d : Matrix
{
public:
	Matrix_d(int rows, int columns);
	virtual matrix getCMatrix() const;
	virtual void allocateMatrixOnDevice();
	virtual void deallocateMatrixOnDevice();
	virtual void uploadMatrixToDevice(Matrix src) const;
	virtual void downloadMatrixFromDevice(Matrix dst) const;
private:
	bool _allocatedToDevice = false;

};

