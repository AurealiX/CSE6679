#ifndef UTIL_H
#define UTIL_H

#include <stddef.h> // For size_t

// Matrix creation/destruction
double** create_double_matrix(size_t rows, size_t cols);

int** create_int_matrix(size_t rows, size_t cols);

void free_matrix(double** matrix, size_t rows);

void free_matrix(int** matrix, size_t rows);

// Matrix initialization
void matrix_set_all(double** matrix, size_t rows, size_t cols, double value);
void matrix_set_all(int** matrix, size_t rows, size_t cols, int value);


// Utility: generate a random double in [min, max].
double randomDouble(double min, double max);

// Copies a vector (1D array)
double* copy_vector(double* src, size_t size);


// Function to print a double** matrix.
void printDoubleMatrix(double **matrix, int rows, int cols);
void printIntMatrix(int **matrix, int rows, int cols);

#endif