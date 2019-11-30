#include "Vector.hpp"
#include <memory>
Vector::Vector(int length, matrixInitialisation initialisation) : Matrix(length, 1, initialisation)
{
	switch (initialisation)
	{
	case matrixInitialisation::M_NO_INIT:
		break;
	case matrixInitialisation::M_ALLOCATE:
		allocate();
		break;
	case matrixInitialisation::M_ASSIGN:
		allocate();
		break;
	default:
		break;
	}
}

Vector::Vector(Matrix src, matrixInitialisation initialisation) : Matrix(src.getRows(), 1, initialisation)
{
	switch (initialisation)
	{
	case matrixInitialisation::M_NO_INIT:
		break;
	case matrixInitialisation::M_ALLOCATE:
		allocate();
		break;
	case matrixInitialisation::M_ASSIGN:
		allocate();
		memcpy(_Cmatrix.elements, src.getCMatrix().elements, sizeof(double) * src.getRows());
		break;
	default:
		break;
	}
}

Vector::~Vector()
{
}
