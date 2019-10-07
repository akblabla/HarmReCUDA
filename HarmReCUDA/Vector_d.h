#pragma once
#include "Matrix_d.h"
#include "Vector.h"
class Vector_d :
	public Matrix_d
{
public:
	Vector_d(int length, matrixInitialisation::matrixInitialisation initialisation = matrixInitialisation::no_init);
	Vector_d(const Vector& src, matrixInitialisation::matrixInitialisation initialisation = matrixInitialisation::no_init);
	virtual ~Vector_d();
};

