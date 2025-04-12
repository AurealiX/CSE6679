#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "transportation.h"
#include "util.h"
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <vector>
#include <algorithm>

// Extern DFS loop search
extern "C" int findLoop(TransportationProblem *problem, int m, int n, int start_row, int start_col, int loop[][2], int *loop_length);

// Dual costs calc
void computeDuals(TransportationProblem *problem, double *u, double *v) {
    int m = problem->numSupply, n = problem->numDemand;
    for (int i = 0; i < m; i++) u[i] = DBL_MAX;
    for (int j = 0; j < n; j++) v[j] = DBL_MAX;
    u[0] = 0;
    bool chg = true;
    while (chg) {
        chg = false;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (problem->BFS[i][j] == 1) {
                    if (u[i] != DBL_MAX && v[j] == DBL_MAX) { v[j] = problem->cost[i][j] - u[i]; chg = true; }
                    if (v[j] != DBL_MAX && u[i] == DBL_MAX) { u[i] = problem->cost[i][j] - v[j]; chg = true; }
                }
            }
        }
    }
}

extern "C"
double ssmGPUSolve(TransportationProblem *problem) {
    // Start timer
    clock_t start = clock();
    int m = problem->numSupply, n = problem->numDemand;
    int iter = 0;
    while (1) {
        iter++;
        // Compute duals
        double *u = new double[m];
        double *v = new double[n];
        computeDuals(problem, u, v);
        double bestDelta = DBL_MAX;
        int best_i = -1, best_j = -1;
        // Find candidate cell
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (problem->BFS[i][j] == 1) continue;
                double d_ij = problem->cost[i][j] - (u[i] + v[j]);
                if (d_ij < bestDelta) { bestDelta = d_ij; best_i = i; best_j = j; }
            }
        }
        delete[] u; delete[] v;
        if (bestDelta >= -1e-10) break;
        // Get loop via DFS
        int loopSize = m + n;
        int (*loop_ptr)[2] = new int[loopSize][2];
        int loop_length = 0;
        if (!findLoop(problem, m, n, best_i, best_j, loop_ptr, &loop_length)) {
            loop_length = 4;
            loop_ptr[0][0] = best_i; loop_ptr[0][1] = best_j;
            loop_ptr[1][0] = best_i; loop_ptr[1][1] = 0;
            loop_ptr[2][0] = 0;      loop_ptr[2][1] = 0;
            loop_ptr[3][0] = 0;      loop_ptr[3][1] = best_j;
        }
        // Compute pivot theta
        double theta = DBL_MAX;
        for (int k = 1; k < loop_length; k += 2) {
            int r = loop_ptr[k][0], c = loop_ptr[k][1];
            if (problem->solution[r][c] < theta)
                theta = problem->solution[r][c];
        }
        if (theta < 1e-10) theta = 1e-8;
        // Pivot update
        int sign = 1;
        for (int k = 0; k < loop_length; k++) {
            int r = loop_ptr[k][0], c = loop_ptr[k][1];
            if (sign > 0)
                problem->solution[r][c] += theta;
            else {
                problem->solution[r][c] -= theta;
                if (fabs(problem->solution[r][c]) < 1e-10) { problem->solution[r][c] = 0; problem->BFS[r][c] = 0; }
            }
            sign = -sign;
        }
        problem->BFS[best_i][best_j] = 1;
        delete[] loop_ptr;
    }
    double elapsed = (double)(clock() - start) / CLOCKS_PER_SEC;
    printf("Pivots: %d, Elapsed: %.4f sec\n", iter, elapsed);
    return elapsed;
}
