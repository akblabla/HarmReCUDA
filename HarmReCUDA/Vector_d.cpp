#include "Vector_d.hpp"
#include <algorithm>

Vector_d::Vector_d(int length, matrixInitialisation initialisation) : Matrix_d(length,1, initialisation)
{
}

Vector_d::Vector_d(const Vector& src, matrixInitialisation initialisation) : Matrix_d(src, initialisation)
{
}

void Vector_d::getSubVector(Vector_d& dest, int rowsStart, int rowsEnd) const
{
	//treat negative numbers as uncapped
	if (rowsEnd < 0) {
		rowsEnd = getRows();
	}
	else {
		rowsEnd++;
	}
	rowsStart = std::min(getRows(), rowsStart);
	rowsEnd = std::min(getRows(), rowsEnd);

	Vector_d result(rowsEnd - rowsStart, AMatrix::M_NO_INIT);
	result._Cmatrix.elements = _Cmatrix.elements + (rowsStart);
	result._Cmatrix.ld = _Cmatrix.ld;
	dest = result;
}

Vector_d::~Vector_d()
{
}
