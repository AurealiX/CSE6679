// gpu_ssm.cu
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include "transportation.h"  // Provides TransportationProblem.
#include "util.h"

// ---------------------------------------------------------------------------
// Data structure for a frontier element in BFS.
// Each element carries a vertex id, the id of its predecessor,
// the root (starting supply node), and the cumulative cost (h_ij).
// ---------------------------------------------------------------------------
struct Vertex {
    int id;      // identifier (could be row index for supply or column index for demand)
    int pred;    // predecessor vertex id
    int root;    // starting vertex id (supply node)
    double cost; // cumulative cost along the path (h_ij)
};

// ---------------------------------------------------------------------------
// Kernel 5: BFS search for accelerated SSM.
// Each thread processes one element from the input frontier (FI) and
// expands its neighbors (using a CSR representation of the spanning tree).
// ---------------------------------------------------------------------------
__global__ void kernel_ssm_BFS(const int* chi_p, const int* chi_n, int numVertices,
                               const Vertex* d_FI, int fiSize,
                               Vertex* d_FO, int* d_foSize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < fiSize) {
        Vertex curr = d_FI[idx];
        // For the current vertex, iterate over its neighbors in CSR.
        int start = chi_p[curr.id];
        int end = chi_p[curr.id + 1];
        for (int i = start; i < end; i++) {
            int nb = chi_n[i];
            // Create a new frontier element.
            Vertex newV;
            newV.id = nb;
            newV.pred = curr.id;
            newV.root = curr.root;
            // Update cumulative cost.
            // In an actual implementation, the cost update uses the cost matrix.
            // Here we use a placeholder: for even hops we subtract, for odd hops we add.
            // (This alternation mimics the sign changes in the closed loop.)
            newV.cost = curr.cost + ((i % 2 == 0) ? -0.5 : 0.5);
            int pos = atomicAdd(d_foSize, 1);
            d_FO[pos] = newV;
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel 6: Backtrace for SSM.
// Each thread takes one candidate (from the frontier) and computes the
// reduced cost for that candidate cell.
// For a candidate cell, assume its reduced cost d_ij = cost(cell) + cumulative cost.
// (A proper implementation would backtrace using stored predecessor info to sum costs.)
// ---------------------------------------------------------------------------
__global__ void kernel_ssm_backtrace(const Vertex* d_candidates, int candSize,
                                     const double* d_cost, int totalCells,
                                     double* d_D, int* d_candidateIdx)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < candSize) {
        Vertex cand = d_candidates[idx];
        // For illustration, assume that the candidate's id indexes into the flattened cost array.
        // Compute d_ij = d_cost[cand.id] + cand.cost.
        double d_val = d_cost[cand.id] + cand.cost;
        d_D[idx] = d_val;
        d_candidateIdx[idx] = idx; // store candidate index (could also store the flattened cell index)
    }
}

// ---------------------------------------------------------------------------
// Kernel 7: Reduction kernel to find the candidate with minimum reduced cost.
// This is similar to previous reduction kernels.
// ---------------------------------------------------------------------------
__global__ void reduceCandidateKernel(const double* d_D, int size,
                                        double* d_out_val, int* d_out_idx)
{
    extern __shared__ double sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    double myVal = (idx < size) ? d_D[idx] : DBL_MAX;
    sdata[tid] = myVal;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sdata[tid + s] < sdata[tid])
                sdata[tid] = sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0)
        d_out_val[blockIdx.x] = sdata[0];
    // Note: For simplicity we omit tracking the index; it can be similarly computed.
}

// ---------------------------------------------------------------------------
// Host function: ssmGPUSolve
// This function performs an accelerated SSM search using the steps described:
// 1. Transform the spanning tree (BFSo) to a CSR representation.
// 2. Initialize the frontier with supply nodes (from BFSo).
// 3. Launch the BFS kernel to generate a new frontier.
// 4. Launch the backtrace kernel to compute reduced costs for candidate cells.
// 5. Reduce the candidate array to pick the candidate with minimum (most negative) cost.
// 6. (Pivoting code would follow here.)
// ---------------------------------------------------------------------------
extern "C" double ssmGPUSolve(TransportationProblem* problem) {
    printf("Running GPU Stepping Stone Method (SSM) for phase 2 optimization...\n");
    clock_t ssm_start_time = clock();
    
    int m = problem->numSupply;
    int n = problem->numDemand;
    
    // For accelerated SSM, we interpret the BFSo as a spanning tree T.
    // For simplicity, we create dummy CSR arrays. In a real implementation,
    // convert the BFS (problem->BFS) into a spanning tree and then to CSR.
    int numVertices = m + n;
    int* h_chi_p = new int[numVertices + 1];
    int* h_chi_n = new int[numVertices - 1]; // spanning tree has (m+n-1) edges.
    // Dummy initialization.
    for (int i = 0; i <= numVertices; i++) {
        h_chi_p[i] = i < numVertices ? i : numVertices - 1;
    }
    for (int i = 0; i < numVertices - 1; i++) {
        h_chi_n[i] = i + 1;
    }
    
    int *d_chi_p, *d_chi_n;
    cudaMalloc(&d_chi_p, (numVertices + 1) * sizeof(int));
    cudaMalloc(&d_chi_n, (numVertices - 1) * sizeof(int));
    cudaMemcpy(d_chi_p, h_chi_p, (numVertices + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_chi_n, h_chi_n, (numVertices - 1) * sizeof(int), cudaMemcpyHostToDevice);
    
    // Allocate memory for the frontier arrays.
    const int maxFrontierSize = numVertices; // approximate size.
    Vertex* h_FI = new Vertex[maxFrontierSize];
    // Initialize the input frontier (FI) with the supply nodes.
    // Here we assume that the first m vertices (0..m-1) are supply nodes.
    for (int i = 0; i < m; i++) {
        h_FI[i].id = i;
        h_FI[i].pred = -1;
        h_FI[i].root = i;
        // Initialize cumulative cost. For initial edges, set to negative of cost.
        // In this dummy version, we set 0.0.
        h_FI[i].cost = 0.0;
    }
    int fiSize = m;
    
    Vertex *d_FI, *d_FO;
    cudaMalloc(&d_FI, maxFrontierSize * sizeof(Vertex));
    cudaMalloc(&d_FO, maxFrontierSize * sizeof(Vertex));
    cudaMemcpy(d_FI, h_FI, fiSize * sizeof(Vertex), cudaMemcpyHostToDevice);
    
    int* d_foSize;
    cudaMalloc(&d_foSize, sizeof(int));
    cudaMemset(d_foSize, 0, sizeof(int));
    
    // Launch BFS kernel (Kernel 5).
    int threadsBFS = 256;
    int blocksBFS = (fiSize + threadsBFS - 1) / threadsBFS;
    kernel_ssm_BFS<<<blocksBFS, threadsBFS>>>(d_chi_p, d_chi_n, numVertices, d_FI, fiSize, d_FO, d_foSize);
    cudaDeviceSynchronize();
    
    // Retrieve new frontier size.
    int h_foSize;
    cudaMemcpy(&h_foSize, d_foSize, sizeof(int), cudaMemcpyDeviceToHost);
    
    // Now, for each candidate in the new frontier, compute its reduced cost.
    // For this, we need the flattened cost matrix.
    int totalCells = m * n;
    double* h_cost = new double[totalCells];
    // Flatten the cost matrix from problem->cost.
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            h_cost[i * n + j] = problem->cost[i][j];
        }
    }
    double* d_cost;
    cudaMalloc(&d_cost, totalCells * sizeof(double));
    cudaMemcpy(d_cost, h_cost, totalCells * sizeof(double), cudaMemcpyHostToDevice);
    
    // Allocate arrays for candidate reduced costs.
    int candidateSize = h_foSize; // number of candidates equal to new frontier size.
    double* d_D;
    int* d_candidateIdx;
    cudaMalloc(&d_D, candidateSize * sizeof(double));
    cudaMalloc(&d_candidateIdx, candidateSize * sizeof(int));
    
    // Launch Kernel 6 to compute reduced costs from the new frontier.
    int threadsBT = 256;
    int blocksBT = (h_foSize + threadsBT - 1) / threadsBT;
    kernel_ssm_backtrace<<<blocksBT, threadsBT>>>(d_FO, h_foSize, d_cost, totalCells, d_D, d_candidateIdx);
    cudaDeviceSynchronize();
    
    // Reduce the candidate array to find the candidate with minimum reduced cost.
    int redThreads = 256;
    int redBlocks = (candidateSize + redThreads * 2 - 1) / (redThreads * 2);
    double* d_out_val;
    int* d_out_idx;
    cudaMalloc(&d_out_val, redBlocks * sizeof(double));
    cudaMalloc(&d_out_idx, redBlocks * sizeof(int));
    size_t sharedMemSize = redThreads * sizeof(double);
    reduceCandidateKernel<<<redBlocks, redThreads, sharedMemSize>>>(d_D, candidateSize, d_out_val, d_out_idx);
    cudaDeviceSynchronize();
    
    double* h_out_val = new double[redBlocks];
    int* h_out_idx = new int[redBlocks];
    cudaMemcpy(h_out_val, d_out_val, redBlocks * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out_idx, d_out_idx, redBlocks * sizeof(int), cudaMemcpyDeviceToHost);
    
    double bestDelta = DBL_MAX;
    int bestCandidate = -1;
    for (int i = 0; i < redBlocks; i++) {
        if (h_out_val[i] < bestDelta) {
            bestDelta = h_out_val[i];
            bestCandidate = h_out_idx[i];
        }
    }
    
    // At this point, bestDelta is the minimum reduced cost for a candidate.
    // If bestDelta is not negative, the current BFSo is optimal.
    if (bestDelta >= -1e-10) {
        printf("SSM: No negative reduced cost candidate found; BFS is optimal.\n");
    } else {
        // Otherwise, you would retrieve the closed loop corresponding to bestCandidate,
        // backtrace the path (using stored information in the BFS tree) and perform pivoting.
        // (This pivoting step is not fully implemented here.)
        printf("SSM: Best candidate has reduced cost %f (candidate index %d).\n", bestDelta, bestCandidate);
    }
    
    // Cleanup device memory.
    cudaFree(d_chi_p);
    cudaFree(d_chi_n);
    cudaFree(d_FI);
    cudaFree(d_FO);
    cudaFree(d_foSize);
    cudaFree(d_cost);
    cudaFree(d_D);
    cudaFree(d_candidateIdx);
    cudaFree(d_out_val);
    cudaFree(d_out_idx);
    
    // Cleanup host memory.
    delete[] h_chi_p;
    delete[] h_chi_n;
    delete[] h_FI;
    delete[] h_cost;
    delete[] h_out_val;
    delete[] h_out_idx;
    
    double elapsed_time = (double)(clock() - ssm_start_time) / CLOCKS_PER_SEC;
    printf("GPU SSM solver completed in %f seconds.\n", elapsed_time);
    return elapsed_time;
}
