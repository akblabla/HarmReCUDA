#include "Vector_d.h"

Vector_d::Vector_d(int length, matrixInitialisation::matrixInitialisation initialisation) : Matrix_d(length,1, initialisation)
{
}

Vector_d::Vector_d(const Vector& src, matrixInitialisation::matrixInitialisation initialisation) : Matrix_d(src, initialisation)
{
}

Vector_d::~Vector_d()
{
}
