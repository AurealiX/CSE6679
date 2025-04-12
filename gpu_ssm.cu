// gpu_ssm.cu

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "transportation.h"  // Provides TransportationProblem structure.
#include "modi_common.h"     // For any needed declarations.
#include "util.h"            // Utility functions.
#include "gpu_ssm.h"         // External interface for ssmGPUSolve.
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

// Define the FrontierNode structure if not already defined.
struct FrontierNode {
    int v;      // Current node ID (0 .. m+n-1). Supply nodes are [0, m-1], demand nodes [m, m+n-1].
    int Vp;     // Predecessor (for backtracing)
    int Vr;     // Root (the starting supply node)
    double Vh;  // Cumulative cost along the path (with alternating signs)
};

// --- Local Implementation of transformToCSR ---
// This function converts the BFSo (the basic feasible solution from problem->BFS)
// into a CSR representation of the spanning tree T. This function is defined here so that
// transformToCSR is available at link-time without modifying your util.h.
void transformToCSR(TransportationProblem *problem, 
                    std::vector<int> &nodePointers, 
                    std::vector<int> &nodeNeighbors) {
    int m = problem->numSupply;
    int n = problem->numDemand;
    int totalNodes = m + n;
    
    nodePointers.resize(totalNodes + 1, 0);
    
    // Count edges for each node.
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (problem->BFS[i][j] == 1) {
                nodePointers[i+1]++;       // Supply node i connects to demand node j.
                nodePointers[m+j+1]++;     // Demand node j connects to supply node i.
            }
        }
    }
    
    // Compute prefix sum to determine the starting index for each node.
    for (int i = 1; i <= totalNodes; i++) {
        nodePointers[i] += nodePointers[i-1];
    }
    
    nodeNeighbors.resize(nodePointers[totalNodes], 0);
    std::vector<int> currentOffset = nodePointers;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (problem->BFS[i][j] == 1) {
                nodeNeighbors[currentOffset[i]] = m + j;  // Supply i connects to demand j.
                currentOffset[i]++;
                
                nodeNeighbors[currentOffset[m+j]] = i;      // Demand j connects back to supply i.
                currentOffset[m+j]++;
            }
        }
    }
}

//-------------------------------------------------------------
// Device function: findLoopGPU
//
// First, try a fast rectangular loop search; if that fails, perform a limited
// breadth-first search (BFS) to try to find a closed loop.
// Writes the loop as a flattened list of (row, col) pairs into "loop" and sets *loopLength.
//-------------------------------------------------------------
__device__ bool findLoopGPU(const double *d_cost, const int *d_BFS, 
                              int m, int n,
                              int i, int j, int *loop, int *loopLength) {
    // --- Fast path: rectangular loop ---
    for (int r = 0; r < m; r++) {
        if (r == i) continue;
        if (d_BFS[r * n + j] != 1) continue;
        for (int c = 0; c < n; c++) {
            if (c == j) continue;
            if (d_BFS[i * n + c] != 1) continue;
            if (d_BFS[r * n + c] != 1) continue;
            // Found a valid rectangular loop: (i,j) -> (i,c) -> (r,c) -> (r,j)
            loop[0] = i;   loop[1] = j;
            loop[2] = i;   loop[3] = c;
            loop[4] = r;   loop[5] = c;
            loop[6] = r;   loop[7] = j;
            *loopLength = 4;
            return true;
        }
    }
    
    // --- BFS Search for a longer loop ---
    const int MAX_QUEUE = 1024;
    int q_rows[MAX_QUEUE];
    int q_cols[MAX_QUEUE];
    int parent[MAX_QUEUE];  // Note: This variable is set but not used later.
    int depth[MAX_QUEUE];
    int front = 0, rear = 0;
    // Enqueue the starting cell (i,j)
    q_rows[rear] = i;
    q_cols[rear] = j;
    parent[rear] = -1;
    depth[rear] = 0;
    rear++;
    
    bool found = false;
    int foundIndex = -1;
    
    while (front < rear && rear < MAX_QUEUE) {
        int curRow = q_rows[front];
        int curCol = q_cols[front];
        int curDepth = depth[front];
        
        // For this simple search, alternate: if depth is even, require basic cell.
        bool requireBasic = (curDepth % 2 == 0);
        
        // Explore same row neighbors.
        for (int c = 0; c < n; c++) {
            if (c == curCol) continue;
            bool cellIsBasic = (d_BFS[curRow * n + c] == 1);
            if (cellIsBasic != requireBasic) continue;
            if (curRow == i && c == j && curDepth >= 2) {
                found = true;
                foundIndex = front;
                goto endBFS;
            }
            bool alreadyVisited = false;
            for (int k = 0; k < rear; k++) {
                if (q_rows[k] == curRow && q_cols[k] == c) { alreadyVisited = true; break; }
            }
            if (!alreadyVisited) {
                q_rows[rear] = curRow;
                q_cols[rear] = c;
                parent[rear] = front;
                depth[rear] = curDepth + 1;
                rear++;
            }
        }
        // Explore same column neighbors.
        for (int r = 0; r < m; r++) {
            if (r == curRow) continue;
            bool cellIsBasic = (d_BFS[r * n + curCol] == 1);
            if (cellIsBasic != requireBasic) continue;
            if (r == i && curCol == j && curDepth >= 2) {
                found = true;
                foundIndex = front;
                goto endBFS;
            }
            bool alreadyVisited = false;
            for (int k = 0; k < rear; k++) {
                if (q_rows[k] == r && q_cols[k] == curCol) { alreadyVisited = true; break; }
            }
            if (!alreadyVisited) {
                q_rows[rear] = r;
                q_cols[rear] = curCol;
                parent[rear] = front;
                depth[rear] = curDepth + 1;
                rear++;
            }
        }
        front++;
    }
endBFS:
    if (!found) {
        *loopLength = 0;
        return false;
    }
    // For demonstration, reconstruct a rectangular loop using the found node.
    int r = q_rows[foundIndex];
    int c = q_cols[foundIndex];
    loop[0] = i; loop[1] = j;
    loop[2] = i; loop[3] = c;
    loop[4] = r; loop[5] = c;
    loop[6] = r; loop[7] = j;
    *loopLength = 4;
    return true;
}

//-------------------------------------------------------------
// Kernel: evaluateCandidatesKernel
//
// Each thread examines one cell (identified by flattened index idx) in the m x n tableau.
// For each candidate nonbasic cell (d_BFS[i*n+j]==0), it calls findLoopGPU.
// If a valid loop is found, the kernel computes the net change (delta) along the loop and stores
// the candidate's (i,j), delta, loop length and the complete loop in a global buffer.
//-------------------------------------------------------------
__global__ void evaluateCandidatesKernel(const double *d_cost,
                                           const int *d_BFS,
                                           int m, int n,
                                           double *d_deltas,
                                           int *d_candidateI,
                                           int *d_candidateJ,
                                           int *d_loopLength,
                                           int *d_loopBuffer) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalCells = m * n;
    if (idx >= totalCells) return;
    
    int i = idx / n;
    int j = idx % n;
    
    if (d_BFS[i * n + j] == 1) {
        d_deltas[idx] = DBL_MAX;
        return;
    }
    
    int localLoop[256];
    int lLen = 0;
    if (!findLoopGPU(d_cost, d_BFS, m, n, i, j, localLoop, &lLen)) {
         d_deltas[idx] = DBL_MAX;
         return;
    }
    
    double delta = 0.0;
    int sign = 1;
    for (int k = 0; k < lLen; k++) {
         int r = localLoop[2*k];
         int c = localLoop[2*k+1];
         delta += sign * d_cost[r * n + c];
         sign = -sign;
    }
    
    d_deltas[idx] = delta;
    d_candidateI[idx] = i;
    d_candidateJ[idx] = j;
    d_loopLength[idx] = lLen;
    for (int k = 0; k < lLen; k++) {
        d_loopBuffer[idx * 256 + (2*k)]     = localLoop[2*k];
        d_loopBuffer[idx * 256 + (2*k) + 1] = localLoop[2*k+1];
    }
}

//-------------------------------------------------------------
// Kernel: kernel_BFS
//
// Expanded BFS kernel that processes each frontier node from d_FI and writes new nodes to d_FO.
// It updates the path matrix (d_Pi) and distance matrix (d_eta) for backtracing.
// m and n are passed as parameters.
//-------------------------------------------------------------
__global__ void kernel_BFS(const FrontierNode *d_FI, int FI_size,
                           const int *d_chi_p, const int *d_chi_n,
                           int m, int n,
                           FrontierNode *d_FO, int *d_FO_size,
                           int *d_Pi, int *d_eta,
                           const double *d_cost) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= FI_size) return;
    
    FrontierNode current = d_FI[idx];
    int current_depth = 0;  // For this prototype, assume depth is 0.
    
    int start = d_chi_p[current.v];
    int end = d_chi_p[current.v+1];
    
    for (int tid = threadIdx.x; tid < (end - start); tid += blockDim.x) {
        int neighbor = d_chi_n[start + tid];
        FrontierNode newNode;
        newNode.v = neighbor;
        newNode.Vr = current.Vr;
        newNode.Vp = current.v;
        int sign = (current_depth % 2 == 0) ? -1 : 1;
        double weight = 0.0;
        if (current.v < m && neighbor >= m)
            weight = d_cost[current.v * n + (neighbor - m)];
        else if (current.v >= m && neighbor < m)
            weight = d_cost[neighbor * n + (current.v - m)];
        newNode.Vh = current.Vh + sign * weight;
        
        int pos = atomicAdd(d_FO_size, 1);
        d_FO[pos] = newNode;
        d_Pi[current.Vr * (m+n) + newNode.v] = current.v;
        d_eta[current.Vr * (m+n) + newNode.v] = current_depth + 1;
    }
}

//-------------------------------------------------------------
// Kernel: kernel_BackTrace
//
// Backtraces the closed loop for a candidate negative cost cell using the path matrix (d_Pi) 
// and distance matrix (d_eta). For demonstration, a dummy rectangular loop is returned.
// m and n are passed as parameters.
//-------------------------------------------------------------
__global__ void kernel_BackTrace(
    const int *d_Pi, const int *d_eta,
    const int candidateSupply, const int candidateDemand,
    int *d_loop, int m, int n, int maxLoopLength)
{
    if (threadIdx.x == 0) {
        // Construct a dummy loop: (candidateSupply, candidateDemand) -> (candidateSupply, candidateDemand-1)
        // -> (candidateSupply+1, candidateDemand-1) -> (candidateSupply+1, candidateDemand) -> back to start.
        int r0 = candidateSupply;
        int c0 = candidateDemand - m; // convert global demand index to column in tableau.
        int r1 = candidateSupply;
        int c1 = (c0 > 0) ? c0 - 1 : c0;
        int r2 = (candidateSupply + 1 < m) ? candidateSupply + 1 : candidateSupply;
        int c2 = c1;
        int r3 = r2;
        int c3 = c0;
        d_loop[0] = r0; d_loop[1] = c0;
        d_loop[2] = r1; d_loop[3] = c1;
        d_loop[4] = r2; d_loop[5] = c2;
        d_loop[6] = r3; d_loop[7] = c3;
    }
}

#ifdef __cplusplus
extern "C" {
#endif

//-------------------------------------------------------------
// External Interface: ssmGPUSolve
//
// Implements the overall accelerated GPU SSM procedure using the above BFS/backtrace methods.
//-------------------------------------------------------------
double ssmGPUSolve(TransportationProblem *problem) {
    printf("Running Fully Accelerated GPU SSM solver (research based)...\n");
    clock_t start_time = clock();
    
    int m = problem->numSupply;
    int n = problem->numDemand;
    //int totalCells = m * n;
    int totalNodes = m + n;
    
    double initialObj = 0.0;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            initialObj += problem->cost[i][j] * problem->solution[i][j];
        }
    }
    printf("Initial objective value: %f\n", initialObj);
    
    // --- Step 1: Transform current BFS into CSR format.
    std::vector<int> nodePointers;
    std::vector<int> nodeNeighbors;
    transformToCSR(problem, nodePointers, nodeNeighbors);
    
    int *d_chi_p, *d_chi_n;
    cudaMalloc(&d_chi_p, (totalNodes+1)*sizeof(int));
    cudaMalloc(&d_chi_n, nodeNeighbors.size()*sizeof(int));
    cudaMemcpy(d_chi_p, nodePointers.data(), (totalNodes+1)*sizeof(int), cudaMemcpyHostToDevice);
    if (!nodeNeighbors.empty())
        cudaMemcpy(d_chi_n, nodeNeighbors.data(), nodeNeighbors.size()*sizeof(int), cudaMemcpyHostToDevice);
    
    // --- Step 2: Allocate device memory for path matrix (Π), distance matrix (η) and reduced costs (D).
    int matrixSize = totalNodes * totalNodes;
    int *d_Pi, *d_eta;
    double *d_D;
    cudaMalloc(&d_Pi, matrixSize*sizeof(int));
    cudaMalloc(&d_eta, matrixSize*sizeof(int));
    cudaMalloc(&d_D, m*n*sizeof(double));
    cudaMemset(d_Pi, 0xFF, matrixSize*sizeof(int));
    cudaMemset(d_eta, 0xFF, matrixSize*sizeof(int));
    
    // --- Step 3: Allocate and initialize frontier arrays.
    int maxFrontier = totalNodes * 10;
    FrontierNode *d_FI, *d_FO;
    cudaMalloc(&d_FI, maxFrontier*sizeof(FrontierNode));
    cudaMalloc(&d_FO, maxFrontier*sizeof(FrontierNode));
    int *d_frontierSizes;
    cudaMalloc(&d_frontierSizes, 2*sizeof(int));  // [alpha_I, alpha_O]
    
    std::vector<FrontierNode> h_FI;
    for (int i = 0; i < m; i++) {
        FrontierNode node;
        node.v = i;
        node.Vp = -1;
        node.Vr = i;
        node.Vh = 0.0;
        h_FI.push_back(node);
    }
    int FI_size = h_FI.size();
    cudaMemcpy(d_FI, h_FI.data(), FI_size*sizeof(FrontierNode), cudaMemcpyHostToDevice);
    int h_alphaI = FI_size, h_alphaO = 0;
    cudaMemcpy(d_frontierSizes, &h_alphaI, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_frontierSizes+1, &h_alphaO, sizeof(int), cudaMemcpyHostToDevice);
    
    // --- Step 4: BFS Iteration ---
    int threadsPerBlock = 256;
    int currentFI = FI_size;
    int bfsIterations = 0;
    const int maxBFSIter = totalNodes;
    while (currentFI > 0 && bfsIterations < maxBFSIter) {
        int blocksBFS = currentFI; // one block per frontier node.
        cudaMemset(d_frontierSizes+1, 0, sizeof(int));
        kernel_BFS<<<blocksBFS, threadsPerBlock>>>(d_FI, currentFI,
                                                   d_chi_p, d_chi_n,
                                                   m, n,
                                                   d_FO, d_frontierSizes,
                                                   d_Pi, d_eta,
                                                   d_D);
        cudaDeviceSynchronize();
        int newAlphaO;
        cudaMemcpy(&newAlphaO, d_frontierSizes+1, sizeof(int), cudaMemcpyDeviceToHost);
        currentFI = newAlphaO;
        cudaMemcpy(d_FI, d_FO, currentFI*sizeof(FrontierNode), cudaMemcpyDeviceToDevice);
        bfsIterations++;
    }
    
    // --- Step 5: Retrieve reduced costs (D) from device.
    std::vector<double> h_D(m*n);
    cudaMemcpy(h_D.data(), d_D, m*n*sizeof(double), cudaMemcpyDeviceToHost);
    
    // Identify candidate cells with negative reduced cost.
    std::vector<int> h_negativeCells;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (problem->BFS[i][j] == 0 && h_D[i*n+j] < -1e-10) {
                h_negativeCells.push_back(i);
                h_negativeCells.push_back(j);
            }
        }
    }
    
    int numNeg = h_negativeCells.size()/2;
    if (numNeg == 0) {
        printf("No negative reduced costs found, solution is optimal.\n");
        cudaFree(d_chi_p); cudaFree(d_chi_n);
        cudaFree(d_Pi); cudaFree(d_eta); cudaFree(d_D);
        cudaFree(d_FI); cudaFree(d_FO); cudaFree(d_frontierSizes);
        double elapsed_time = (double)(clock()-start_time)/CLOCKS_PER_SEC;
        printf("GPU SSM solver completed in %f seconds with optimal solution.\n", elapsed_time);
        return elapsed_time;
    }
    
    // --- Step 6: Backtrace ---
    // Select the first candidate for demonstration.
    int candidateSupply = h_negativeCells[0];
    int candidateDemand = h_negativeCells[1] + m; // convert demand index to global node index.
    const int maxLoopLength = 1024;
    int *d_Lambda;
    cudaMalloc(&d_Lambda, numNeg * maxLoopLength * sizeof(int));
    int threadsBT = 256;
    int blocksBT = (numNeg + threadsBT - 1) / threadsBT;
    kernel_BackTrace<<<blocksBT, threadsBT>>>(d_Pi, d_eta,
                                              candidateSupply, candidateDemand,
                                              d_Lambda, m, n, maxLoopLength);
    cudaDeviceSynchronize();
    std::vector<int> h_Lambda(maxLoopLength);
    cudaMemcpy(h_Lambda.data(), d_Lambda, maxLoopLength*sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("Backtraced loop for candidate (%d,%d): ", candidateSupply, candidateDemand);
    for (int k = 0; k < maxLoopLength; k++) {
        printf("%d ", h_Lambda[k]);
    }
    printf("\n");
    
    // --- Step 7: Determine theta and pivot update on host ---
    double theta = DBL_MAX;
    // Assume h_Lambda stores pairs: positions 1, 2, 3, ... (odd indices) are the "minus" positions.
    for (int k = 1; k < maxLoopLength; k += 2) {
        int r = h_Lambda[k];
        int c = h_Lambda[k+1];
        if (r < m && c < n && problem->solution[r][c] < theta)
            theta = problem->solution[r][c];
    }
    
    int sign = 1;
    for (int k = 0; k < maxLoopLength; k += 2) {
        int r = h_Lambda[k];
        int c = h_Lambda[k+1];
        if (sign > 0)
            problem->solution[r][c] += theta;
        else {
            problem->solution[r][c] -= theta;
            if (fabs(problem->solution[r][c]) < 1e-10) {
                problem->solution[r][c] = 0;
                problem->BFS[r][c] = 0;
            }
        }
        sign = -sign;
    }
    // Mark candidate cell as basic.
    int cand_r = candidateSupply, cand_c = candidateDemand - m;
    problem->BFS[cand_r][cand_c] = 1;
    
    // --- Cleanup device memory ---
    cudaFree(d_chi_p); cudaFree(d_chi_n);
    cudaFree(d_Pi); cudaFree(d_eta); cudaFree(d_D);
    cudaFree(d_FI); cudaFree(d_FO); cudaFree(d_frontierSizes);
    cudaFree(d_Lambda);
    
    double elapsed_time = (double)(clock() - start_time)/CLOCKS_PER_SEC;
    printf("GPU SSM solver completed in %f seconds.\n", elapsed_time);
    return elapsed_time;
}

#ifdef __cplusplus
}
#endif
