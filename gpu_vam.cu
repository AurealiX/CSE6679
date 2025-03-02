#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <cuda_runtime.h>
#include "gpu_vam.h"
#include "transportation.h"
#include "util.h"

// For timing
#include <time.h>

// -----------------------------------------------------------------------------
// Kernel to compute row penalties.
// Each thread computes the penalty for one row (if not already marked done).
// The cost matrix is assumed to be stored in row-major order.
__global__ void computeRowPenaltiesKernel(const double* cost, int numSupply, int numDemand,
                                            const int* col_done, const int* row_done,
                                            double* row_penalty, int* row_min_index) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < numSupply) {
        if (row_done[row]) {
            row_penalty[row] = -1;
            row_min_index[row] = -1;
            return;
        }
        double min1 = FLT_MAX;
        double min2 = FLT_MAX;
        int minIndex = -1;
        for (int j = 0; j < numDemand; j++) {
            if (col_done[j]) continue;
            double cost_ij = cost[row * numDemand + j];
            if (cost_ij < min1) {
                min2 = min1;
                min1 = cost_ij;
                minIndex = j;
            } else if (cost_ij < min2) {
                min2 = cost_ij;
            }
        }
        row_penalty[row] = (min2 == FLT_MAX) ? min1 : (min2 - min1);
        row_min_index[row] = minIndex;
    }
}

// -----------------------------------------------------------------------------
// Kernel to compute column penalties.
// Each thread computes the penalty for one column (if not already marked done).
__global__ void computeColPenaltiesKernel(const double* cost, int numSupply, int numDemand,
                                            const int* row_done, const int* col_done,
                                            double* col_penalty, int* col_min_index) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < numDemand) {
        if (col_done[col]) {
            col_penalty[col] = -1;
            col_min_index[col] = -1;
            return;
        }
        double min1 = FLT_MAX;
        double min2 = FLT_MAX;
        int minIndex = -1;
        for (int i = 0; i < numSupply; i++) {
            if (row_done[i]) continue;
            double cost_ij = cost[i * numDemand + col];
            if (cost_ij < min1) {
                min2 = min1;
                min1 = cost_ij;
                minIndex = i;
            } else if (cost_ij < min2) {
                min2 = cost_ij;
            }
        }
        col_penalty[col] = (min2 == FLT_MAX) ? min1 : (min2 - min1);
        col_min_index[col] = minIndex;
    }
}

// -----------------------------------------------------------------------------
// gpuVamSolve: Implements GPU-accelerated Vogel's Approximation Method.
// It follows the structure of the CPU version, but offloads penalty computations.
double gpuVamSolve(TransportationProblem* problem) {
    clock_t start = clock();

    int numSupply = problem->numSupply;
    int numDemand = problem->numDemand;
    int totalCells = numSupply * numDemand;

    // Allocate and copy cost matrix to device (assumed row-major order).
    double* h_cost = (double*)malloc(totalCells * sizeof(double));
    for (int i = 0; i < numSupply; i++) {
        for (int j = 0; j < numDemand; j++) {
            h_cost[i * numDemand + j] = problem->cost[i][j];
        }
    }
    double* d_cost;
    cudaMalloc((void**)&d_cost, totalCells * sizeof(double));
    cudaMemcpy(d_cost, h_cost, totalCells * sizeof(double), cudaMemcpyHostToDevice);
    free(h_cost);

    // Allocate and initialize row_done and col_done arrays (host and device).
    int* h_row_done = (int*)calloc(numSupply, sizeof(int));
    int* h_col_done = (int*)calloc(numDemand, sizeof(int));
    int* d_row_done;
    int* d_col_done;
    cudaMalloc((void**)&d_row_done, numSupply * sizeof(int));
    cudaMalloc((void**)&d_col_done, numDemand * sizeof(int));
    cudaMemcpy(d_row_done, h_row_done, numSupply * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_done, h_col_done, numDemand * sizeof(int), cudaMemcpyHostToDevice);

    // Create copies of the supply and demand arrays on the host.
    double* remainingSupply = copy_vector(problem->supply, numSupply);
    double* remainingDemand = copy_vector(problem->demand, numDemand);

    // Allocate device arrays for row and column penalties.
    double* d_row_penalty;
    int* d_row_min_index;
    cudaMalloc((void**)&d_row_penalty, numSupply * sizeof(double));
    cudaMalloc((void**)&d_row_min_index, numSupply * sizeof(int));

    double* d_col_penalty;
    int* d_col_min_index;
    cudaMalloc((void**)&d_col_penalty, numDemand * sizeof(double));
    cudaMalloc((void**)&d_col_min_index, numDemand * sizeof(int));

    // Kernel launch configuration.
    int blockSize = 128;
    int gridSizeRow = (numSupply + blockSize - 1) / blockSize;
    int gridSizeCol = (numDemand + blockSize - 1) / blockSize;

    int rows_remaining = numSupply;
    int cols_remaining = numDemand;

    // Main loop: iterate until either all rows or columns are allocated.
    while (rows_remaining > 0 && cols_remaining > 0) {
        // Launch kernels to compute row and column penalties.
        computeRowPenaltiesKernel<<<gridSizeRow, blockSize>>>(d_cost, numSupply, numDemand,
                                                              d_col_done, d_row_done,
                                                              d_row_penalty, d_row_min_index);
        cudaDeviceSynchronize();

        computeColPenaltiesKernel<<<gridSizeCol, blockSize>>>(d_cost, numSupply, numDemand,
                                                              d_row_done, d_col_done,
                                                              d_col_penalty, d_col_min_index);
        cudaDeviceSynchronize();

        // Copy penalty results back to host.
        double* h_row_penalty = (double*)malloc(numSupply * sizeof(double));
        int* h_row_min_index = (int*)malloc(numSupply * sizeof(int));
        cudaMemcpy(h_row_penalty, d_row_penalty, numSupply * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_row_min_index, d_row_min_index, numSupply * sizeof(int), cudaMemcpyDeviceToHost);

        double* h_col_penalty = (double*)malloc(numDemand * sizeof(double));
        int* h_col_min_index = (int*)malloc(numDemand * sizeof(int));
        cudaMemcpy(h_col_penalty, d_col_penalty, numDemand * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_col_min_index, d_col_min_index, numDemand * sizeof(int), cudaMemcpyDeviceToHost);

        // Determine the maximum penalty among rows and columns.
        double max_penalty = -1;
        int selectedRow = -1;
        int selectedCol = -1;
        // Check rows.
        for (int i = 0; i < numSupply; i++) {
            if (!h_row_done[i] && h_row_penalty[i] > max_penalty) {
                max_penalty = h_row_penalty[i];
                selectedRow = i;
                selectedCol = h_row_min_index[i];
            }
        }
        // Check columns.
        for (int j = 0; j < numDemand; j++) {
            if (!h_col_done[j] && h_col_penalty[j] > max_penalty) {
                max_penalty = h_col_penalty[j];
                selectedCol = j;
                selectedRow = h_col_min_index[j];
            }
        }
        free(h_row_penalty);
        free(h_row_min_index);
        free(h_col_penalty);
        free(h_col_min_index);

        if (selectedRow == -1 || selectedCol == -1) {
            // No valid allocation can be made.
            break;
        }

        // Determine allocation amount.
        double allocation = (remainingSupply[selectedRow] < remainingDemand[selectedCol])
                            ? remainingSupply[selectedRow] : remainingDemand[selectedCol];

        // Update solution and BFS matrix on the host.
        problem->solution[selectedRow][selectedCol] = allocation;
        problem->BFS[selectedRow][selectedCol] = 1;

        remainingSupply[selectedRow] -= allocation;
        remainingDemand[selectedCol] -= allocation;

        // Mark row/column as done if fully allocated.
        if (remainingSupply[selectedRow] <= 1e-10) {
            h_row_done[selectedRow] = 1;
            rows_remaining--;
        }
        if (remainingDemand[selectedCol] <= 1e-10) {
            h_col_done[selectedCol] = 1;
            cols_remaining--;
        }
        // Copy updated row_done and col_done back to device.
        cudaMemcpy(d_row_done, h_row_done, numSupply * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_col_done, h_col_done, numDemand * sizeof(int), cudaMemcpyHostToDevice);
    }

    // Free resources.
    free(h_row_done);
    free(h_col_done);
    free(remainingSupply);
    free(remainingDemand);
    cudaFree(d_cost);
    cudaFree(d_row_done);
    cudaFree(d_col_done);
    cudaFree(d_row_penalty);
    cudaFree(d_row_min_index);
    cudaFree(d_col_penalty);
    cudaFree(d_col_min_index);

    double elapsed_time = (double)(clock() - start) / CLOCKS_PER_SEC;
    printf("GPU VAM solver completed in %f seconds.\n", elapsed_time);
    return elapsed_time;
}