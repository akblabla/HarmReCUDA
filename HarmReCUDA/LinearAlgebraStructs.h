#pragma once
#define MATRIX_INDEX(row,column,rowCount) row + column*rowCount
struct matrix {
	double* elements;
	int rows;
	int columns;
	int ld;
};