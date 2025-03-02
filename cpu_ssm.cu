#include "cpu_ssm.h"
#include "transportation.h"
#include "modi_common.h"
#include "util.h"
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <string.h>
#include <time.h>

// The Stepping Stone Method (SSM) implementation for Phase 2 optimization.
// For each nonbasic cell, it finds a closed loop (using DFS via findLoop),
// computes the net (alternating) cost delta along that loop,
// and then pivots along the loop corresponding to the most negative delta.
// The process is repeated until no negative delta is found.
double ssmCPUSolve(TransportationProblem *problem) {
    printf("Running Stepping Stone Method (SSM) for phase 2 optimization...\n");
    clock_t ssm_start_time = clock();

    int m = problem->numSupply;
    int n = problem->numDemand;
    int optimal = 0;

    // Continue iterating until no improvement (negative delta) is found.
    while (!optimal) {
        double bestDelta = 0.0;  // best (most negative) improvement found
        int best_i = -1, best_j = -1;
        // bestLoop will hold the closed-loop path for the candidate cell.
        int bestLoop[m+n][2];
        int bestLoopLength = 0;

        // Step 1: Evaluate each nonbasic cell.
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (problem->BFS[i][j] == 1)
                    continue;  // skip basic cells

                int loop[m+n][2];
                int loopLength = 0;
                // Attempt to find a closed loop starting from (i,j).
                if (!findLoop(problem, m, n, i, j, loop, &loopLength))
                    continue; // no closed loop found

                // Compute the net cost change (delta) along the loop.
                // In SSM, assign a plus sign to the candidate cell (first cell)
                // and alternate signs for subsequent cells.
                double delta = 0.0;
                int sign = 1;
                for (int k = 0; k < loopLength; k++) {
                    int r = loop[k][0];
                    int c = loop[k][1];
                    delta += sign * problem->cost[r][c];
                    sign = -sign;
                }

                // Update the best candidate if delta is more negative.
                if (delta < bestDelta ||
                    (fabs(delta - bestDelta) < 1e-10 && (i < best_i || (i == best_i && j < best_j)))) {
                    bestDelta = delta;
                    best_i = i;
                    best_j = j;
                    bestLoopLength = loopLength;
                    memcpy(bestLoop, loop, loopLength * 2 * sizeof(int));
                }
            }
        }

        // If no candidate with negative delta is found, the solution is optimal.
        if (best_i == -1 || bestDelta >= -1e-10) {
            optimal = 1;
            break;
        }

        // Step 2: Determine the maximum feasible adjustment (theta).
        // In the closed loop, the candidate cell (best_i, best_j) is at index 0 (plus sign).
        // The cells in odd positions (indices 1, 3, …) have negative signs.
        double theta = DBL_MAX;
        for (int k = 1; k < bestLoopLength; k += 2) {
            int r = bestLoop[k][0];
            int c = bestLoop[k][1];
            if (problem->solution[r][c] < theta)
                theta = problem->solution[r][c];
        }

        // Step 3: Adjust allocations along the loop.
        // Add theta to the candidate cell and subtract theta from cells with negative sign.
        int sign = 1;
        for (int k = 0; k < bestLoopLength; k++) {
            int r = bestLoop[k][0];
            int c = bestLoop[k][1];
            if (sign > 0) {
                problem->solution[r][c] += theta;
            } else {
                problem->solution[r][c] -= theta;
                // If allocation becomes zero, remove it from the basis.
                if (fabs(problem->solution[r][c]) < 1e-10) {
                    problem->solution[r][c] = 0;
                    problem->BFS[r][c] = 0;
                }
            }
            sign = -sign;
        }
        // Mark the candidate cell as basic.
        problem->BFS[best_i][best_j] = 1;
    }

    double elapsed_time = (double)(clock() - ssm_start_time) / CLOCKS_PER_SEC;
    printf("CPU SSM solver completed in %f seconds.\n", elapsed_time);
    return elapsed_time;
}
