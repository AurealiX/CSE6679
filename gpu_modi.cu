#include "cpu_modi.h"      // provides TransportationProblem, findLoop(), etc.
#include "modi_common.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <string.h>

// Define tile dimensions for the reduced cost kernel.
#define TILE_SIZE 16

// Sentinel value used in GPU dual-potential computation for an unknown potential.
#define UNKNOWN (-1e20)

// Custom atomicExch for doubles using atomicCAS.
// This function atomically exchanges the value at address with val and returns the old value.
__device__ double atomicExchDouble(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*) address;
    unsigned long long int old = *address_as_ull;
    unsigned long long int assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val));
    } while (assumed != old);
    return __longlong_as_double(old);
}

//-----------------------------------------------------------------------------
// Kernel: computeReducedCostKernel
// Each thread computes one cell: D[i][j] = cost[i][j] - u[i] - v[j]
//-----------------------------------------------------------------------------
__global__ void computeReducedCostKernel(const double *d_cost, 
                                           const double *d_u, 
                                           const double *d_v, 
                                           double *d_D, 
                                           int m, int n) 
{
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    if (row < m && col < n) {
        double u_val = d_u[row];
        double v_val = d_v[col];
        d_D[row * n + col] = d_cost[row * n + col] - u_val - v_val;
    }
}

//-----------------------------------------------------------------------------
// Kernel: reduceMinKernel
// This kernel uses shared memory to reduce the flattened array d_D to find the
// minimum value and its flattened index.
//-----------------------------------------------------------------------------
__global__ void reduceMinKernel(const double *d_D, int size, 
                                double *d_out_val, int *d_out_idx) 
{
    extern __shared__ char sdata[];
    double* s_val = (double*) sdata;
    int* s_idx = (int*)(sdata + blockDim.x * sizeof(double));

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    
    double minVal;
    int minIdx;
    
    if (i < size) {
        minVal = d_D[i];
        minIdx = i;
        if (i + blockDim.x < size) {
            double temp = d_D[i + blockDim.x];
            if (temp < minVal) {
                minVal = temp;
                minIdx = i + blockDim.x;
            }
        }
    } else {
        minVal = DBL_MAX;
        minIdx = -1;
    }
    s_val[tid] = minVal;
    s_idx[tid] = minIdx;
    __syncthreads();
    
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_val[tid + s] < s_val[tid]) {
                s_val[tid] = s_val[tid + s];
                s_idx[tid] = s_idx[tid + s];
            }
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        d_out_val[blockIdx.x] = s_val[0];
        d_out_idx[blockIdx.x] = s_idx[0];
    }
}

//-----------------------------------------------------------------------------
// Kernel: updatePotentialsKernel
// Each block processes one supply row and each thread one demand column.
// For each basic cell (BFS==1), if one potential is known and the other unknown,
// update the unknown potential. A flag d_changed is set if any update occurs.
//-----------------------------------------------------------------------------
__global__ void updatePotentialsKernel(const double *d_cost, const int *d_BFS, 
                                         double *d_u, double *d_v,
                                         int m, int n, int *d_changed) {
    int i = blockIdx.x;      // one block per supply row
    int j = threadIdx.x;     // one thread per demand column in that row
    if (i < m && j < n) {
        int idx = i * n + j;
        if (d_BFS[idx] == 1) { // if cell (i,j) is basic
            double u_val = d_u[i];
            double v_val = d_v[j];
            // If u[i] is known and v[j] is unknown, update v[j]
            if (u_val > UNKNOWN + 1e-5 && v_val == UNKNOWN) {
                double new_v = d_cost[idx] - u_val;
                atomicExchDouble(&d_v[j], new_v);
                *d_changed = 1;
            }
            // If v[j] is known and u[i] is unknown, update u[i]
            else if (v_val > UNKNOWN + 1e-5 && u_val == UNKNOWN) {
                double new_u = d_cost[idx] - v_val;
                atomicExchDouble(&d_u[i], new_u);
                *d_changed = 1;
            }
        }
    }
}

//-----------------------------------------------------------------------------
// Function: computePotentialsGPU
// Computes the dual potentials (u and v) on the GPU using iterative relaxation.
// d_cost: flattened cost matrix (size m*n)
// d_BFS: flattened basic solution indicator (size m*n)
// d_u: device array for supply potentials (length m)
// d_v: device array for demand potentials (length n)
void computePotentialsGPU(const double *d_cost, const int *d_BFS, 
                          double *d_u, double *d_v, int m, int n) {
    int *d_changed;
    cudaMalloc((void**)&d_changed, sizeof(int));
    int h_changed = 0;
    
    // Initialize potentials on host and copy to device.
    double *h_u = (double*)malloc(m * sizeof(double));
    double *h_v = (double*)malloc(n * sizeof(double));
    for (int i = 0; i < m; i++) {
        h_u[i] = UNKNOWN;
    }
    for (int j = 0; j < n; j++) {
        h_v[j] = UNKNOWN;
    }
    h_u[0] = 0.0;  // set an arbitrary starting potential
    cudaMemcpy(d_u, h_u, m * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v, n * sizeof(double), cudaMemcpyHostToDevice);
    free(h_u);
    free(h_v);
    
    // Launch kernel iteratively until no changes occur.
    dim3 grid(m, 1, 1);
    dim3 block(n, 1, 1);  // one thread per demand column in each row
    do {
        h_changed = 0;
        cudaMemcpy(d_changed, &h_changed, sizeof(int), cudaMemcpyHostToDevice);
        updatePotentialsKernel<<<grid, block>>>(d_cost, d_BFS, d_u, d_v, m, n, d_changed);
        cudaDeviceSynchronize();
        cudaMemcpy(&h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost);
    } while (h_changed);
    cudaFree(d_changed);
}

#ifdef __cplusplus
extern "C" {
#endif

//-----------------------------------------------------------------------------
// Function: modiGPUSolve
// GPU-accelerated MODI solver for Phase 2 optimization.
// Offloads dual potential computation to the GPU.
//-----------------------------------------------------------------------------
double modiGPUSolve(TransportationProblem *problem) {
    // Start overall timing.
    printf("Running MODI Method on GPU for phase 2 optimization...\n");
    clock_t modi_start_time = clock();

    int m = problem->numSupply;
    int n = problem->numDemand;
    
    // Allocate host memory for flattened cost matrix.
    size_t costSize = m * n * sizeof(double);
    double *h_cost = (double*)malloc(costSize);
    for (int i = 0; i < m; i++) {
        memcpy(&h_cost[i * n], problem->cost[i], n * sizeof(double));
    }
    
    // Allocate device memory for cost matrix, potentials, and reduced cost matrix.
    double *d_cost, *d_u, *d_v, *d_D;
    cudaMalloc((void**)&d_cost, costSize);
    cudaMalloc((void**)&d_u, m * sizeof(double));
    cudaMalloc((void**)&d_v, n * sizeof(double));
    cudaMalloc((void**)&d_D, costSize);
    
    cudaMemcpy(d_cost, h_cost, costSize, cudaMemcpyHostToDevice);
    
    // Allocate device memory for BFS matrix.
    size_t bfsSize = m * n * sizeof(int);
    int *d_BFS;
    cudaMalloc((void**)&d_BFS, bfsSize);
    
    // Setup grid and block dimensions for computeReducedCostKernel.
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((n + TILE_SIZE - 1) / TILE_SIZE, (m + TILE_SIZE - 1) / TILE_SIZE);
    
    // Preallocate reduction buffers.
    int size = m * n;
    int threads = 256;
    int blocks = (size + threads * 2 - 1) / (threads * 2);
    double *d_out_val;
    int *d_out_idx;
    cudaMalloc((void**)&d_out_val, blocks * sizeof(double));
    cudaMalloc((void**)&d_out_idx, blocks * sizeof(int));
    double *h_out_val = (double*)malloc(blocks * sizeof(double));
    int *h_out_idx = (int*)malloc(blocks * sizeof(int));
    
    // Create CUDA events for kernel timing (used internally, not printed).
    cudaEvent_t eventStart, eventStop;
    cudaEventCreate(&eventStart);
    cudaEventCreate(&eventStop);
    
    int optimal = 0;
    while (!optimal) {
        // Flatten the host BFS (problem->BFS, an int**) into a contiguous array.
        int *h_BFS_flat = (int*)malloc(bfsSize);
        for (int i = 0; i < m; i++) {
            memcpy(&h_BFS_flat[i * n], problem->BFS[i], n * sizeof(int));
        }
        cudaMemcpy(d_BFS, h_BFS_flat, bfsSize, cudaMemcpyHostToDevice);
        free(h_BFS_flat);
        
        // Compute dual potentials on GPU.
        computePotentialsGPU(d_cost, d_BFS, d_u, d_v, m, n);
        
        // Compute the reduced cost matrix on GPU.
        cudaEventRecord(eventStart, 0);
        computeReducedCostKernel<<<gridDim, blockDim>>>(d_cost, d_u, d_v, d_D, m, n);
        cudaDeviceSynchronize();
        cudaEventRecord(eventStop, 0);
        cudaEventSynchronize(eventStop);
        
        // Launch reduction kernel.
        size_t sharedMemSize = threads * (sizeof(double) + sizeof(int));
        cudaEventRecord(eventStart, 0);
        reduceMinKernel<<<blocks, threads, sharedMemSize>>>(d_D, size, d_out_val, d_out_idx);
        cudaDeviceSynchronize();
        cudaEventRecord(eventStop, 0);
        cudaEventSynchronize(eventStop);
        
        // Copy reduction results to host.
        cudaMemcpy(h_out_val, d_out_val, blocks * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_out_idx, d_out_idx, blocks * sizeof(int), cudaMemcpyDeviceToHost);
        
        double bestDelta = DBL_MAX;
        int best_index = -1;
        for (int i = 0; i < blocks; i++) {
            if (h_out_val[i] < bestDelta) {
                bestDelta = h_out_val[i];
                best_index = h_out_idx[i];
            }
        }
        
        int best_i = best_index / n;
        int best_j = best_index % n;
        
        // Check for optimality.
        if (bestDelta >= -1e-10) {
            optimal = 1;
            break;
        }
        
        // On the host: find a closed loop (cycle) for the candidate cell.
        int loop_max = m + n;
        int (*loop)[2] = new int[loop_max][2];
        int loop_length = 0;
        if (!findLoop(problem, m, n, best_i, best_j, loop, &loop_length)) {
            delete[] loop;
            break;
        }
        
        // Determine theta: the minimum allocation among negative cells in the loop.
        double theta = DBL_MAX;
        int sign = -1;
        for (int k = 1; k < loop_length; k++) {
            if (sign < 0) {
                int r = loop[k][0];
                int c = loop[k][1];
                if (problem->solution[r][c] < theta)
                    theta = problem->solution[r][c];
            }
            sign = -sign;
        }
        
        // Pivot: adjust allocations along the loop.
        sign = 1;
        for (int k = 0; k < loop_length; k++) {
            int r = loop[k][0];
            int c = loop[k][1];
            if (sign > 0)
                problem->solution[r][c] += theta;
            else {
                if (fabs(problem->solution[r][c] - theta) < 1e-10)
                    problem->BFS[r][c] = 0;
                problem->solution[r][c] -= theta;
            }
            sign = -sign;
        }
        // Mark the candidate cell as basic.
        problem->BFS[best_i][best_j] = 1;
        if (problem->solution[best_i][best_j] < 1e-10)
            problem->solution[best_i][best_j] += 1e-10;
        
        // Free the dynamically allocated loop memory.
        delete[] loop;
    }
    
    // Cleanup device memory.
    cudaFree(d_cost);
    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_D);
    cudaFree(d_BFS);
    cudaFree(d_out_val);
    cudaFree(d_out_idx);
    
    // Cleanup host memory.
    free(h_cost);
    free(h_out_val);
    free(h_out_idx);
    
    cudaEventDestroy(eventStart);
    cudaEventDestroy(eventStop);
    
    double elapsed_time = (double)(clock() - modi_start_time) / CLOCKS_PER_SEC;
    return elapsed_time;
}

#ifdef __cplusplus
}
#endif
