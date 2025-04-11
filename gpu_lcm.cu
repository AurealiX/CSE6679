// gpu_lcm.cu
#include "transportation.h"  // Provides TransportationProblem definition.
#include "util.h"            // Contains helper functions such as copy_vector() and matrix utilities.
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <float.h>
#include <ctime>
#include <cstring>

// Structure to hold a candidate (cell) for allocation.
struct MinCell {
    double cost;
    int row;
    int col;
};

// Kernel #1: Compute candidates
// Each thread examines one cell in the flattened cost matrix. If the corresponding
// supply row and demand column are not marked as done, the candidate cost is the cost;
// otherwise, it is set to DBL_MAX.
__global__ void computeCandidateKernel(const double* d_cost, const int* d_row_done, 
                                         const int* d_col_done, int m, int n, 
                                         MinCell* d_candidates)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = m * n;
    if (idx < total) {
        int i = idx / n;
        int j = idx % n;
        MinCell cand;
        if (d_row_done[i] == 0 && d_col_done[j] == 0) {
            cand.cost = d_cost[idx];
            cand.row = i;
            cand.col = j;
        }
        else {
            cand.cost = DBL_MAX;
            cand.row = -1;
            cand.col = -1;
        }
        d_candidates[idx] = cand;
    }
}

// Kernel #2: Reduction kernel for candidate array.
// Uses shared memory to reduce the candidate array to a single cell with the minimum cost.
__global__ void reduceMinCellKernel(MinCell* d_in, MinCell* d_out, int nElements)
{
    extern __shared__ MinCell sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    
    MinCell myMin;
    myMin.cost = DBL_MAX;
    myMin.row = -1;
    myMin.col = -1;
    
    if (i < nElements) {
        myMin = d_in[i];
        if (i + blockDim.x < nElements) {
            MinCell other = d_in[i + blockDim.x];
            if (other.cost < myMin.cost)
                myMin = other;
        }
    }
    sdata[tid] = myMin;
    __syncthreads();
    
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sdata[tid + s].cost < sdata[tid].cost)
                sdata[tid] = sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        d_out[blockIdx.x] = sdata[0];
    }
}

// Host helper function: Perform full reduction on candidate array to obtain the minimum cell.
MinCell reduceMinCell(MinCell* d_candidates, int numElements)
{
    int threads = 256;
    int blocks = (numElements + threads * 2 - 1) / (threads * 2);
    
    MinCell* d_intermediate;
    cudaMalloc(&d_intermediate, blocks * sizeof(MinCell));
    
    reduceMinCellKernel<<<blocks, threads, threads * sizeof(MinCell)>>>(d_candidates, d_intermediate, numElements);
    cudaDeviceSynchronize();
    
    int s = blocks;
    while (s > 1) {
        int threadsNew = 256;
        int blocksNew = (s + threadsNew * 2 - 1) / (threadsNew * 2);
        reduceMinCellKernel<<<blocksNew, threadsNew, threadsNew * sizeof(MinCell)>>>(d_intermediate, d_intermediate, s);
        cudaDeviceSynchronize();
        s = blocksNew;
    }
    
    MinCell h_min;
    cudaMemcpy(&h_min, d_intermediate, sizeof(MinCell), cudaMemcpyDeviceToHost);
    cudaFree(d_intermediate);
    return h_min;
}

// GPU implementation of the Least Cost Method (LCM) in C++.
// This function calculates the elapsed time using clock() in the same format as the MODI method.
extern "C" double lcmGPUSolve(TransportationProblem *problem)
{
    printf("Running GPU Least Cost Method (Phase 1) in C++...\n");
    clock_t lcm_start_time = clock();
    
    int m = problem->numSupply;
    int n = problem->numDemand;
    int totalCells = m * n;
    
    // Allocate and flatten the cost matrix.
    double* h_cost = new double[totalCells];
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            h_cost[i * n + j] = problem->cost[i][j];
        }
    }
    
    // Allocate and copy supply and demand vectors.
    double* h_supply = new double[m];
    memcpy(h_supply, problem->supply, m * sizeof(double));
    double* h_demand = new double[n];
    memcpy(h_demand, problem->demand, n * sizeof(double));
    
    // Device memory allocations.
    double *d_cost, *d_supply, *d_demand, *d_solution;
    int *d_row_done, *d_col_done;
    cudaMalloc(&d_cost, totalCells * sizeof(double));
    cudaMalloc(&d_supply, m * sizeof(double));
    cudaMalloc(&d_demand, n * sizeof(double));
    cudaMalloc(&d_solution, totalCells * sizeof(double));
    cudaMalloc(&d_row_done, m * sizeof(int));
    cudaMalloc(&d_col_done, n * sizeof(int));
    
    // Copy data to device.
    cudaMemcpy(d_cost, h_cost, totalCells * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_supply, h_supply, m * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_demand, h_demand, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemset(d_solution, 0, totalCells * sizeof(double));
    cudaMemset(d_row_done, 0, m * sizeof(int));
    cudaMemset(d_col_done, 0, n * sizeof(int));
    
    // Allocate device candidate array.
    MinCell* d_candidates;
    cudaMalloc(&d_candidates, totalCells * sizeof(MinCell));
    
    // Keep track of uncovered rows and columns.
    int rows_remaining = m;
    int cols_remaining = n;
    
    // Main iterative loop (at most m+n-1 iterations).
    while (rows_remaining > 0 && cols_remaining > 0) {
        int threads = 256;
        int blocks = (totalCells + threads - 1) / threads;
        computeCandidateKernel<<<blocks, threads>>>(d_cost, d_row_done, d_col_done, m, n, d_candidates);
        cudaDeviceSynchronize();
        
        MinCell minCell = reduceMinCell(d_candidates, totalCells);
        if (minCell.row == -1 || minCell.col == -1) {
            // No valid cell found.
            break;
        }
        
        // Allocate the minimum of remaining supply and demand.
        double allocation = (h_supply[minCell.row] < h_demand[minCell.col])
                            ? h_supply[minCell.row]
                            : h_demand[minCell.col];
                            
        int index = minCell.row * n + minCell.col;
        cudaMemcpy(d_solution + index, &allocation, sizeof(double), cudaMemcpyHostToDevice);
        
        // Update the host copies.
        h_supply[minCell.row] -= allocation;
        h_demand[minCell.col] -= allocation;
        
        // Mark row or column as exhausted.
        if (h_supply[minCell.row] <= 1e-10) {
            int done = 1;
            cudaMemcpy(d_row_done + minCell.row, &done, sizeof(int), cudaMemcpyHostToDevice);
            rows_remaining--;
        }
        if (h_demand[minCell.col] <= 1e-10) {
            int done = 1;
            cudaMemcpy(d_col_done + minCell.col, &done, sizeof(int), cudaMemcpyHostToDevice);
            cols_remaining--;
        }
        
        // Update device supply/demand (optional, for consistency).
        cudaMemcpy(d_supply + minCell.row, &h_supply[minCell.row], sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_demand + minCell.col, &h_demand[minCell.col], sizeof(double), cudaMemcpyHostToDevice);
    }
    
    // Copy the final solution from device back to the problem's solution matrix.
    double* h_solution = new double[totalCells];
    cudaMemcpy(h_solution, d_solution, totalCells * sizeof(double), cudaMemcpyDeviceToHost);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            problem->solution[i][j] = h_solution[i * n + j];
        }
    }
    
    // Cleanup host memory.
    delete[] h_cost;
    delete[] h_supply;
    delete[] h_demand;
    delete[] h_solution;
    
    // Cleanup device memory.
    cudaFree(d_cost);
    cudaFree(d_supply);
    cudaFree(d_demand);
    cudaFree(d_solution);
    cudaFree(d_row_done);
    cudaFree(d_col_done);
    cudaFree(d_candidates);
    
    double elapsed_time = (double)(clock() - lcm_start_time) / CLOCKS_PER_SEC;
    printf("GPU LCM solver completed in %f seconds.\n", elapsed_time);
    
    return elapsed_time;
}
