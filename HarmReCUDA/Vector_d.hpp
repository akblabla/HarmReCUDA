#pragma once
#include "Matrix_d.hpp"
#include "Vector.hpp"
extern class Vector_d :
	public Matrix_d
{
public:
	Vector_d(int length, matrixInitialisation initialisation = matrixInitialisation::M_NO_INIT);
	Vector_d(const Vector& src, matrixInitialisation initialisation = matrixInitialisation::M_NO_INIT);
	virtual void getSubVector(Vector_d& dest, int rowsStart, int rowsEnd) const;
	virtual ~Vector_d();
};

