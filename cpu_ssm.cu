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
#include <omp.h>
#include <vector>
#include <array>

double ssmCPUSolve(TransportationProblem *problem) {
    printf("Running Accelerated Stepping Stone Method (SSM) for phase 2 optimization (CPU accelerated)...\n");
    clock_t start_time = clock();

    int m = problem->numSupply;
    int n = problem->numDemand;
    int optimal = 0;

    // Use a vector to hold the best candidate loop.
    // Maximum loop length is (m+n).
    std::vector<std::array<int, 2>> bestLoop(m + n);
    int bestLoopLength = 0;

    while (!optimal) {
        double bestDelta = 0.0;  // We seek a negative improvement.
        int best_i = -1, best_j = -1;

        // Parallelize over all nonbasic cells.
        #pragma omp parallel for collapse(2) schedule(dynamic)
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (problem->BFS[i][j] == 1)
                    continue;  // Skip basic cells.

                // Each thread uses its own temporary storage for a candidate loop.
                std::vector<std::array<int, 2>> localLoop(m + n);
                int loopLength = 0;

                // Call findLoop with a pointer cast from localLoop.data().
                // Since std::array<int,2> is standard-layout, we can reinterpret_cast it to an int (*)[2].
                if (!findLoop(problem, m, n, i, j, reinterpret_cast<int (*)[2]>(localLoop.data()), &loopLength))
                    continue; // No closed loop found.

                // Compute the net cost change (delta) along the loop.
                double delta = 0.0;
                int sign = 1;
                for (int k = 0; k < loopLength; k++) {
                    int r = localLoop[k][0];
                    int c = localLoop[k][1];
                    delta += sign * problem->cost[r][c];
                    sign = -sign;
                }

                // Use a critical section to update the best candidate if needed.
                #pragma omp critical
                {
                    if (delta < bestDelta ||
                        (fabs(delta - bestDelta) < 1e-10 &&
                         (best_i == -1 || i < best_i || (i == best_i && j < best_j)))) {
                        bestDelta = delta;
                        best_i = i;
                        best_j = j;
                        bestLoopLength = loopLength;
                        for (int k = 0; k < loopLength; k++) {
                            bestLoop[k] = localLoop[k];
                        }
                    }
                }
            }
        }

        // Terminate if no candidate with negative delta is found.
        if (best_i == -1 || bestDelta >= -1e-10) {
            optimal = 1;
            break;
        }

        // Determine the maximum feasible adjustment (theta) along the loop.
        double theta = DBL_MAX;
        for (int k = 1; k < bestLoopLength; k += 2) { // Only positions with negative sign.
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
                // Remove from basis if allocation becomes effectively zero.
                if (fabs(problem->solution[r][c]) < 1e-10) {
                    problem->solution[r][c] = 0;
                    problem->BFS[r][c] = 0;
                }
            }
            sign = -sign;
        }
        // Mark the candidate cell as basic.
        problem->BFS[best_i][best_j] = 1;
    } // end while

    double elapsed_time = (double)(clock() - start_time) / CLOCKS_PER_SEC;
    printf("Accelerated CPU SSM solver completed in %f seconds.\n", elapsed_time);
    return elapsed_time;
}
