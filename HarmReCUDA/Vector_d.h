#pragma once
#include "Matrix_d.h"
#include "Vector.h"
class Vector_d :
	public Matrix_d
{
public:
	Vector_d(int length, matrixInitialisation initialisation = matrixInitialisation::M_NO_INIT);
	Vector_d(const Vector& src, matrixInitialisation initialisation = matrixInitialisation::M_NO_INIT);
	virtual ~Vector_d();
};

