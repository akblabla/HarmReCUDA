#include "mat.h"
#include "matlabInterface.hpp"
#include <string>
#include <stdio.h>
#include <stdlib.h>



int diagnose(const char* file) {
	MATFile* pmat;
	const char** dir;
	const char* name;
	int	  ndir;
	int	  i;
	mxArray* pa;

	printf("Reading file %s...\n\n", file);

	/*
	 * Open file to get directory
	 */
	pmat = matOpen(file, "r");
	if (pmat == NULL) {
		printf("Error opening file %s\n", file);
		return(1);
	}

	/*
	 * get directory of MAT-file
	 */
	dir = (const char**)matGetDir(pmat, &ndir);
	if (dir == NULL) {
		printf("Error reading directory of file %s\n", file);
		return(1);
	}
	else {
		printf("Directory of %s:\n", file);
		for (i = 0; i < ndir; i++)
			printf("%s\n", dir[i]);
	}
	mxFree(dir);

	/* In order to use matGetNextXXX correctly, reopen file to read in headers. */
	if (matClose(pmat) != 0) {
		printf("Error closing file %s\n", file);
		return(1);
	}
	pmat = matOpen(file, "r");
	if (pmat == NULL) {
		printf("Error reopening file %s\n", file);
		return(1);
	}

	/* Get headers of all variables */
	printf("\nExamining the header for each variable:\n");
	for (i = 0; i < ndir; i++) {
		pa = matGetNextVariableInfo(pmat, &name);
		if (pa == NULL) {
			printf("Error reading in file %s\n", file);
			return(1);
		}
		/* Diagnose header pa */
		printf("According to its header, array %s has %d dimensions\n",
			name, mxGetNumberOfDimensions(pa));
		if (mxIsFromGlobalWS(pa))
			printf("  and was a global variable when saved\n");
		else
			printf("  and was a local variable when saved\n");
		mxDestroyArray(pa);
	}

	/* Reopen file to read in actual arrays. */
	if (matClose(pmat) != 0) {
		printf("Error closing file %s\n", file);
		return(1);
	}
	pmat = matOpen(file, "r");
	if (pmat == NULL) {
		printf("Error reopening file %s\n", file);
		return(1);
	}

	/* Read in each array. */
	printf("\nReading in the actual array contents:\n");
	for (i = 0; i < ndir; i++) {
		pa = matGetNextVariable(pmat, &name);
		if (pa == NULL) {
			printf("Error reading in file %s\n", file);
			return(1);
		}
		/*
		 * Diagnose array pa
		 */
		printf("According to its contents, array %s has %d dimensions\n",
			name, mxGetNumberOfDimensions(pa));
		if (mxIsFromGlobalWS(pa))
			printf("  and was a global variable when saved\n");
		else
			printf("  and was a local variable when saved\n");
		mxDestroyArray(pa);
	}

	if (matClose(pmat) != 0) {
		printf("Error closing file %s\n", file);
		return(1);
	}
	printf("Done\n");
	return(0);
}

Matrix matLoad(const char* file)
{
	// open MAT-file
	MATFile* pmat = matOpen(file, "r");
	if (pmat == NULL) {
		printf("no file has been found\n");
		Matrix dest(10, 10, Matrix::M_ALLOCATE);
		return dest;
	}
	int	  ndir;
	/*
   * get directory of MAT-file
   */
	auto dir = (const char**)matGetDir(pmat, &ndir);
	if (dir == NULL) {
		printf("Error reading directory of file %s\n", file);
	}
	else {
		//printf("Directory of %s:\n", file);
		//for (int i = 0; i < ndir; i++)
		//	printf("%s\n", dir[i]);
	}
	mxFree(dir);

	// extract the specified variable
	mxArray* arr = matGetVariable(pmat, "data");
	if (arr != NULL && mxIsDouble(arr) && !mxIsEmpty(arr)) {
		// copy data
		mwSize rows = mxGetM(arr);
		mwSize columns = mxGetN(arr);
		double* pr = mxGetPr(arr);
		if (pr != NULL) {
			Matrix dest(rows,columns,Matrix::M_ALLOCATE);
			memcpy(dest.getCMatrix().elements, pr, sizeof(double) * dest.getElementsCount());
			mxDestroyArray(arr);
			matClose(pmat);
			return dest;
		}
	}

	// cleanup
	mxDestroyArray(arr);
	matClose(pmat);
	printf("loading failed\n");
	Matrix dest(10, 10, Matrix::M_ALLOCATE);
	return dest;
}

void matSave(const char* file, Matrix& source)
{
	// open MAT-file
	MATFile* pmat = matOpen(file, "wL");
	if (pmat == NULL) {
		printf("no file has been found\n");
		return;
	}
	int	  ndir;
	/*
   * get directory of MAT-file
   */
	/*
	auto dir = (const char**)matGetDir(pmat, &ndir);
	if (dir == NULL) {
		printf("Error reading directory of file %s\n", file);
	}
	else {
		//printf("Directory of %s:\n", file);
		//for (int i = 0; i < ndir; i++)
		//	printf("%s\n", dir[i]);
	}
	mxFree(dir);*/

	// extract the specified variable
	mxArray* arr = mxCreateDoubleMatrix(source.getRows(), source.getColumns(), mxREAL);

	double* pr = mxGetPr(arr);
	memcpy(pr, source.getCMatrix().elements, sizeof(double) * source.getElementsCount());

	matPutVariable(pmat, "data", arr);

	// cleanup
	mxDestroyArray(arr);
	matClose(pmat);
}
