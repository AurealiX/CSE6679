#include "util.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Creates a 2D matrix (uninitialized values)
double** create_double_matrix(size_t rows, size_t cols) {
    double** matrix = (double**)malloc(rows * sizeof(double*));
    if (!matrix) return NULL;

    for (size_t i = 0; i < rows; i++) {
        matrix[i] = (double*)malloc(cols * sizeof(double));
        if (!matrix[i]) {
            // Cleanup on allocation failure
            for (size_t j = 0; j < i; j++) free(matrix[j]);
            free(matrix);
            return NULL;
        }
    }
    return matrix;
}

int** create_int_matrix(size_t rows, size_t cols) {
    int** matrix = (int**)malloc(rows * sizeof(int*));
    if (!matrix) return NULL;

    for (size_t i = 0; i < rows; i++) {
        matrix[i] = (int*)malloc(cols * sizeof(int));
        if (!matrix[i]) {
            // Cleanup on allocation failure
            for (size_t j = 0; j < i; j++) free(matrix[j]);
            free(matrix);
            return NULL;
        }
    }
    return matrix;
}

// Frees a 2D matrix
void free_matrix(double** matrix, size_t rows) {
    if (!matrix) return;
    
    for (size_t i = 0; i < rows; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

void free_matrix(int** matrix, size_t rows) {
    if (!matrix) return;
    
    for (size_t i = 0; i < rows; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

// Sets all matrix elements to a specific value
void matrix_set_all(double** matrix, size_t rows, size_t cols, double value) {
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            matrix[i][j] = value;
        }
    }
}

void matrix_set_all(int** matrix, size_t rows, size_t cols, int value) {
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            matrix[i][j] = value;
        }
    }
}

// Utility: generate a random double in [min, max].
double randomDouble(double min, double max) {
    return min + ((double)rand() / RAND_MAX) * (max - min);
}

// Copies a vector (1D array)
double* copy_vector(double* src, size_t size) {
    if (!src || size == 0) return NULL;

    // Allocate memory for the new vector
    double* dest = (double*)malloc(size * sizeof(double));
    if (!dest) return NULL;

    // Copy elements from src to dest
    memcpy(dest, src, size * sizeof(double));

    return dest;
}


// Function to print an int** matrix.
void printIntMatrix(int **matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%d ", matrix[i][j]);
        }
        printf("\n");
    }
}

// Function to print a double** matrix.
void printDoubleMatrix(double **matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%lf ", matrix[i][j]);
        }
        printf("\n");
    }
}