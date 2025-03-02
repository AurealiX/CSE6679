#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "transportation.h"
#include "util.h"
#include "cpu_vam.h"
#include "cpu_modi.h"
#include "gpu_vam.h"

TransportationProblem* duplicateTransportationProblem(TransportationProblem* prob);

int main() {
    // Parameters for the random instance
    int num_supply = 100;
    int num_demand = 57;
    int max_cost = 10;   // Maximum cost value for each cell
    int seed = 1;
    printf("Enter seed: ");
    scanf("%d", &seed);

    printf("Enter number of supply nodes: ");
    scanf("%d", &num_supply);

    printf("Enter number of demand nodes: ");
    scanf("%d", &num_demand);

    int max_supply = num_supply + num_demand; // Maximum supply value per source

    // Generate a random test instance for CPU
    TransportationProblem* cpu_problem = createTransportationProblem(num_supply, num_demand);
    generateRandomInstance(cpu_problem, num_supply, num_demand, max_supply, 0, max_cost, 1);

    // Duplicate the instance for GPU so both solvers start with identical data.
    TransportationProblem* gpu_problem = duplicateTransportationProblem(cpu_problem);

    // --- CPU VAM and MODI ---
    printf("Running CPU VAM solver...\n");
    double cpu_vam_time = vamCPUSolve(cpu_problem);
    printf("CPU VAM solve completed in: %f seconds.\n", cpu_vam_time);

    printf("CPU initial solution feasibility is %s, the objective value is %f \n",
           isFeasible(cpu_problem) ? "true" : "false", getObjectiveValue(cpu_problem));

    double cpu_modi_time = modiCPUSolve(cpu_problem);
    printf("CPU Phase 2 optimization completed in: %f seconds.\n", cpu_modi_time);
    printf("CPU final solution feasibility is %s, the objective value is %f \n",
           isFeasible(cpu_problem) ? "true" : "false", getObjectiveValue(cpu_problem));

    // --- GPU VAM and MODI ---
    printf("\nRunning GPU VAM solver...\n");
    double gpu_vam_time = gpuVamSolve(gpu_problem);
    printf("GPU VAM solve completed in: %f seconds.\n", gpu_vam_time);

    // For Phase 2 on the GPU instance, we use the same CPU MODI function.
    double gpu_modi_time = modiCPUSolve(gpu_problem);
    printf("GPU problem Phase 2 optimization (using CPU MODI) completed in: %f seconds.\n", gpu_modi_time);
    printf("GPU problem final solution feasibility is %s, the objective value is %f \n",
           isFeasible(gpu_problem) ? "true" : "false", getObjectiveValue(gpu_problem));

    // --- Compare the two solutions ---
    int match = 1;
    double tolerance = 1e-6;
    for (int i = 0; i < num_supply; i++) {
        for (int j = 0; j < num_demand; j++) {
            if (fabs(cpu_problem->solution[i][j] - gpu_problem->solution[i][j]) > tolerance) {
                match = 0;
                break;
            }
        }
        if (!match) break;
    }
    if (match) {
        printf("CPU and GPU solutions match.\n");
    } else {
        printf("CPU and GPU solutions do NOT match.\n");
    }

    // Cleanup resources.
    // freeTransportationProblem(cpu_problem);
    // freeTransportationProblem(gpu_problem);

    return 0;
}
