#pragma once
#define MATRIX_INDEX(row,column,m) m.transposed ? (column) + (row)*m.ld : (row) + (column)*m.ld
struct matrix {
	double* elements;
	int rows;
	int columns;
	int ld;
	bool transposed;
};