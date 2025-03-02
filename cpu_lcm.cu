#include "cpu_lcm.h"
#include "transportation.h"
#include "util.h"
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <time.h>

double lcmCPUSolve(TransportationProblem *problem) {
    printf("Running Least Cost Method (Phase 1)...\n");
    clock_t lcm_start_time = clock();

    int m = problem->numSupply;
    int n = problem->numDemand;

    // Arrays to mark rows/columns that are fully allocated.
    int *row_done = (int*)calloc(m, sizeof(int));
    int *col_done = (int*)calloc(n, sizeof(int));
    int rows_remaining = m;
    int cols_remaining = n;

    // Copy supply and demand values.
    double *remainingSupply = copy_vector(problem->supply, m);
    double *remainingDemand = copy_vector(problem->demand, n);

    // Loop until every row or every column is exhausted.
    while (rows_remaining > 0 && cols_remaining > 0) {
        double min_cost = DBL_MAX;
        int selectedRow = -1;
        int selectedCol = -1;

        // Scan through all uncovered cells.
        for (int i = 0; i < m; i++) {
            if (row_done[i]) continue;
            for (int j = 0; j < n; j++) {
                if (col_done[j]) continue;
                double cost_ij = problem->cost[i][j];
                if (cost_ij < min_cost) {
                    min_cost = cost_ij;
                    selectedRow = i;
                    selectedCol = j;
                }
            }
        }

        // If no valid cell is found, break out of the loop.
        if (selectedRow == -1 || selectedCol == -1) {
            break;
        }

        // Allocate the minimum of the remaining supply and demand.
        double allocation = (remainingSupply[selectedRow] < remainingDemand[selectedCol])
                            ? remainingSupply[selectedRow]
                            : remainingDemand[selectedCol];
        problem->solution[selectedRow][selectedCol] = allocation;
        problem->BFS[selectedRow][selectedCol] = 1;

        // Update the remaining supply and demand.
        remainingSupply[selectedRow] -= allocation;
        remainingDemand[selectedCol] -= allocation;

        // Mark the row or column as done if its remaining amount is (almost) zero.
        if (remainingSupply[selectedRow] <= 1e-10) {
            row_done[selectedRow] = 1;
            rows_remaining--;
        }
        if (remainingDemand[selectedCol] <= 1e-10) {
            col_done[selectedCol] = 1;
            cols_remaining--;
        }
    }

    free(row_done);
    free(col_done);
    free(remainingSupply);
    free(remainingDemand);

    double elapsed_time = (double)(clock() - lcm_start_time) / CLOCKS_PER_SEC;
    printf("CPU LCM solver completed in %f seconds.\n", elapsed_time);
    return elapsed_time;
}
