#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "transportation.h"
#include "util.h"
#include "cpu_vam.h"
#include "cpu_modi.h"
#include "gpu_modi.h"
#include "cpu_lcm.h"
#include "cpu_ssm.h"  // Include the SSM header

// Helper function to clone a TransportationProblem instance.
TransportationProblem* cloneTransportationProblem(TransportationProblem* original) {
    int m = original->numSupply;
    int n = original->numDemand;
    TransportationProblem* copy = createTransportationProblem(m, n);
    
    // Copy supply and demand vectors.
    memcpy(copy->supply, original->supply, m * sizeof(double));
    memcpy(copy->demand, original->demand, n * sizeof(double));
    
    // Copy cost, solution, and BFS matrices.
    for (int i = 0; i < m; i++) {
        memcpy(copy->cost[i], original->cost[i], n * sizeof(double));
        memcpy(copy->solution[i], original->solution[i], n * sizeof(double));
        memcpy(copy->BFS[i], original->BFS[i], n * sizeof(int));
    }
    
    return copy;
}

int main() {
    int num_supply = 100;    // default values
    int num_demand = 57;
    int max_cost = 10;
    int seed = 1;
    
    // Input parameters.
    printf("Enter seed: ");
    scanf("%d", &seed);
    printf("Enter number of supply nodes: ");
    scanf("%d", &num_supply);
    printf("Enter number of demand nodes: ");
    scanf("%d", &num_demand);

    int max_supply = num_supply + num_demand; // Maximum supply value per source

    // Generate a random test instance.
    TransportationProblem* problem = createTransportationProblem(num_supply, num_demand);
    generateRandomInstance(problem, num_supply, num_demand, max_supply, 0, max_cost, seed);
    
    // --- Phase 1: Compute the initial BFS using VAM (or LCM) ---
    double vamTime = vamCPUSolve(problem);
    printf("VAM initial solution in: %f seconds\n", vamTime);
    printf("VAM solution feasibility is %s, objective value: %f\n",
           isFeasible(problem) ? "true" : "false", getObjectiveValue(problem));
    
    // --- Clone the BFS produced in Phase 1 for independent tests ---
    
    // Test LCM (should produce a BFS similar to VAM when given the same input).
    TransportationProblem* problem_lcm = cloneTransportationProblem(problem);
    double lcmTime = lcmCPUSolve(problem_lcm);
    printf("LCM initial solution in: %f seconds\n", lcmTime);
    printf("LCM solution feasibility is %s, objective value: %f\n",
           isFeasible(problem_lcm) ? "true" : "false", getObjectiveValue(problem_lcm));
    destroyTransportationProblem(problem_lcm);

    // Test Phase 2 using CPU MODI.
    TransportationProblem* problem_cpu = cloneTransportationProblem(problem);
    double cpuTime = modiCPUSolve(problem_cpu);
    printf("Phase 2 optimization (CPU MODI) in: %f seconds\n", cpuTime);
    printf("After CPU MODI, solution feasibility is %s, objective value: %f\n",
           isFeasible(problem_cpu) ? "true" : "false", getObjectiveValue(problem_cpu));
    destroyTransportationProblem(problem_cpu);

    // Test Phase 2 using GPU MODI.
    TransportationProblem* problem_gpu = cloneTransportationProblem(problem);
    double gpuTime = modiGPUSolve(problem_gpu);
    printf("Phase 2 optimization (GPU MODI) in: %f seconds\n", gpuTime);
    printf("After GPU MODI, solution feasibility is %s, objective value: %f\n",
           isFeasible(problem_gpu) ? "true" : "false", getObjectiveValue(problem_gpu));
    destroyTransportationProblem(problem_gpu);

    // Test Phase 2 using CPU SSM.
    TransportationProblem* problem_ssm = cloneTransportationProblem(problem);
    double ssmTime = ssmCPUSolve(problem_ssm);
    printf("Phase 2 optimization (CPU SSM) in: %f seconds\n", ssmTime);
    printf("After CPU SSM, solution feasibility is %s, objective value: %f\n",
           isFeasible(problem_ssm) ? "true" : "false", getObjectiveValue(problem_ssm));
    destroyTransportationProblem(problem_ssm);

    // Clean up the original instance.
    destroyTransportationProblem(problem);
    
    return 0;
}
