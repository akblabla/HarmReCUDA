#pragma once
#define MATRIX_INDEX(row,column,rowCount) rowCount * column + row
struct matrix {
	double* elements;
	int rows;
	int columns;
};