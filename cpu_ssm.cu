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

double ssmCPUSolve(TransportationProblem *problem) {
    printf("Running Stepping Stone Method (SSM) for phase 2 optimization...\n");
    clock_t ssm_start_time = clock();

    int m = problem->numSupply;
    int n = problem->numDemand;
    int optimal = 0;

    // Dynamically allocate 2D arrays with (m+n) rows and 2 columns.
    int (*bestLoop)[2] = new int[m+n][2];
    int bestLoopLength = 0;
    int (*loop)[2] = new int[m+n][2];

    // Continue iterating until no improvement (negative delta) is found.
    while (!optimal) {
        double bestDelta = 0.0;  // best (most negative) improvement found
        int best_i = -1, best_j = -1;

        // Evaluate each nonbasic cell.
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (problem->BFS[i][j] == 1)
                    continue;  // skip basic cells

                int loopLength = 0;
                // Attempt to find a closed loop starting from (i,j).
                if (!findLoop(problem, m, n, i, j, loop, &loopLength))
                    continue; // no closed loop found

                // Compute the net cost change (delta) along the loop.
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

        // Check for termination: if no candidate with negative delta is found.
        if (best_i == -1 || bestDelta >= -1e-10) {
            optimal = 1;
            break;
        }

        // Determine the maximum feasible adjustment (theta).
        double theta = DBL_MAX;
        for (int k = 1; k < bestLoopLength; k += 2) {
            int r = bestLoop[k][0];
            int c = bestLoop[k][1];
            if (problem->solution[r][c] < theta)
                theta = problem->solution[r][c];
        }

        // Adjust allocations along the loop.
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

    // Free the dynamic memory.
    delete[] bestLoop;
    delete[] loop;

    double elapsed_time = (double)(clock() - ssm_start_time) / CLOCKS_PER_SEC;
    printf("CPU SSM solver completed in %f seconds.\n", elapsed_time);
    return elapsed_time;
}
