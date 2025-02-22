#include "cpu_vam.h"
#include "transportation.h"
#include "util.h"
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <time.h>

// VAM algorithm CPU implementation.
// This function computes an initial feasible solution using Vogel's Approximation Method.
// Return the execution time
double vamCPUSolve(TransportationProblem *problem) {
    printf("Running Vogel's Approximation Method (Phase 1)...\n");
    // Record the start time.
    clock_t vam_start_time = clock();

    int m = problem->numSupply;
    int n = problem->numDemand;
    
    // Create arrays to mark completed rows/columns.
    int *row_done = (int*)calloc(m, sizeof(int));  // 0 = not done, 1 = done.
    int *col_done = (int*)calloc(n, sizeof(int));
    int rows_remaining = m;
    int cols_remaining = n;

    double* remainingSupply=copy_vector(problem->supply, m);
    double* remainingDemand=copy_vector(problem->demand,n);
    
    // Loop until all rows or all columns are exhausted.
    while (rows_remaining > 0 && cols_remaining > 0) {
        // Compute penalties for each row.
        double *row_penalty = (double*)malloc(m * sizeof(double));
        int *row_min_index = (int*)malloc(m * sizeof(int));
        for (int i = 0; i < m; i++) {
            if (row_done[i]) {
                row_penalty[i] = -1;
                continue;
            }
            double min1 = FLT_MAX, min2 = FLT_MAX;
            int min_index = -1;
            for (int j = 0; j < n; j++) {
                if (col_done[j]) continue;
                double cost_ij = problem->cost[i][j];
                if (cost_ij < min1) {
                    min2 = min1;
                    min1 = cost_ij;
                    min_index = j;
                } else if (cost_ij < min2) {
                    min2 = cost_ij;
                }
            }
            if (min_index == -1) {
                row_penalty[i] = -1;
            } else {
                // If only one valid cell exists, set penalty to min1.
                row_penalty[i] = (min2 == FLT_MAX) ? min1 : (min2 - min1);
            }
            row_min_index[i] = min_index;
        }
        
        // Compute penalties for each column.
        double *col_penalty = (double*)malloc(n * sizeof(double));
        int *col_min_index = (int*)malloc(n * sizeof(int));
        for (int j = 0; j < n; j++) {
            if (col_done[j]) {
                col_penalty[j] = -1;
                continue;
            }
            double min1 = FLT_MAX, min2 = FLT_MAX;
            int min_index = -1;
            for (int i = 0; i < m; i++) {
                if (row_done[i]) continue;
                double cost_ij = problem->cost[i][j];
                if (cost_ij < min1) {
                    min2 = min1;
                    min1 = cost_ij;
                    min_index = i;
                } else if (cost_ij < min2) {
                    min2 = cost_ij;
                }
            }
            if (min_index == -1) {
                col_penalty[j] = -1;
            } else {
                col_penalty[j] = (min2 == FLT_MAX) ? min1 : (min2 - min1);
            }
            col_min_index[j] = min_index;
        }
        
        // Select the row or column with the maximum penalty.
        double max_penalty = -1;
        int selectedRow = -1, selectedCol = -1;
        // Check rows.
        for (int i = 0; i < m; i++) {
            if (!row_done[i] && row_penalty[i] > max_penalty) {
                max_penalty = row_penalty[i];
                selectedRow = i;
                selectedCol = row_min_index[i];
            }
        }
        // Check columns.
        for (int j = 0; j < n; j++) {
            if (!col_done[j] && col_penalty[j] > max_penalty) {
                max_penalty = col_penalty[j];
                selectedCol = j;
                selectedRow = col_min_index[j];
            }
        }
        
        free(row_penalty);
        free(row_min_index);
        free(col_penalty);
        free(col_min_index);
        
        if (selectedRow == -1 || selectedCol == -1) {
            // No valid allocation can be made.
            break;
        }
        
        // Determine allocation at the selected cell.
        double allocation = (remainingSupply[selectedRow] < remainingDemand[selectedCol])
                           ? remainingSupply[selectedRow] : remainingDemand[selectedCol];
        problem->solution[selectedRow][selectedCol] = allocation;
        problem->BFS[selectedRow][selectedCol]=1;
        // Update supply and demand.
        remainingSupply[selectedRow] -= allocation;
        remainingDemand[selectedCol] -= allocation;
        
        // Mark row or column as done if fully allocated.
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

    // Calculate and print the elapsed time.
    double elapsed_time = (double)(clock() - vam_start_time) / CLOCKS_PER_SEC;
    printf("CPU VAM solver completed in %f seconds.\n", elapsed_time);
    return elapsed_time;
}
