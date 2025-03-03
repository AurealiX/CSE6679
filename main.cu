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
#include "cpu_ssm.h"
#include "gpu_vam.h"  // Header for GPU VAM

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

    // --- Phase 1: Run all initial BFS solvers on fresh instances ---
    // Create a base instance for CPU VAM.
    TransportationProblem* instance_cpu_vam = createTransportationProblem(num_supply, num_demand);
    generateRandomInstance(instance_cpu_vam, num_supply, num_demand, max_supply, 0, max_cost, seed);
    double vamTime = vamCPUSolve(instance_cpu_vam);
    //printf("CPU VAM initial solution in: %f seconds\n", vamTime);
    printf("CPU VAM solution feasibility is %s, objective value: %f\n",
           isFeasible(instance_cpu_vam) ? "true" : "false", getObjectiveValue(instance_cpu_vam));
    destroyTransportationProblem(instance_cpu_vam);

    // Create a fresh instance for GPU VAM.
    TransportationProblem* instance_gpu_vam = createTransportationProblem(num_supply, num_demand);
    generateRandomInstance(instance_gpu_vam, num_supply, num_demand, max_supply, 0, max_cost, seed);
    double gpuVamTime = gpuVamSolve(instance_gpu_vam);
    //printf("GPU VAM initial solution in: %f seconds\n", gpuVamTime);
    printf("GPU VAM solution feasibility is %s, objective value: %f\n",
           isFeasible(instance_gpu_vam) ? "true" : "false", getObjectiveValue(instance_gpu_vam));
    destroyTransportationProblem(instance_gpu_vam);

    // Create a fresh instance for LCM.
    TransportationProblem* instance_lcm = createTransportationProblem(num_supply, num_demand);
    generateRandomInstance(instance_lcm, num_supply, num_demand, max_supply, 0, max_cost, seed);
    double lcmTime = lcmCPUSolve(instance_lcm);
    //printf("LCM initial solution in: %f seconds\n", lcmTime);
    printf("LCM solution feasibility is %s, objective value: %f\n",
           isFeasible(instance_lcm) ? "true" : "false", getObjectiveValue(instance_lcm));
    destroyTransportationProblem(instance_lcm);
    
    // --- Phase 2: Use the BFS from a base instance for all Phase 2 tests ---
    // We choose CPU VAM as our Phase 1 solver to produce the BFS for Phase 2.
    TransportationProblem* base_problem = createTransportationProblem(num_supply, num_demand);
    generateRandomInstance(base_problem, num_supply, num_demand, max_supply, 0, max_cost, seed);
    // Compute the BFS using CPU VAM.
    vamCPUSolve(base_problem);

    // Now clone the BFS-computed instance for each Phase 2 test.

    // Test Phase 2 using CPU MODI.
    TransportationProblem* modi_instance = cloneTransportationProblem(base_problem);
    double cpuModiTime = modiCPUSolve(modi_instance);
    //printf("Phase 2 optimization (CPU MODI) in: %f seconds\n", cpuModiTime);
    printf("After CPU MODI, solution feasibility is %s, objective value: %f\n",
           isFeasible(modi_instance) ? "true" : "false", getObjectiveValue(modi_instance));
    destroyTransportationProblem(modi_instance);

    // Test Phase 2 using GPU MODI.
    TransportationProblem* gpuModi_instance = cloneTransportationProblem(base_problem);
    double gpuModiTime = modiGPUSolve(gpuModi_instance);
    printf("Phase 2 optimization (GPU MODI) in: %f seconds\n", gpuModiTime);
    printf("After GPU MODI, solution feasibility is %s, objective value: %f\n",
           isFeasible(gpuModi_instance) ? "true" : "false", getObjectiveValue(gpuModi_instance));
    destroyTransportationProblem(gpuModi_instance);

    // Test Phase 2 using CPU Stepping Stone Method (SSM).
    TransportationProblem* ssm_instance = cloneTransportationProblem(base_problem);
    double ssmTime = ssmCPUSolve(ssm_instance);
    printf("Phase 2 optimization (CPU SSM) in: %f seconds\n", ssmTime);
    printf("After CPU SSM, solution feasibility is %s, objective value: %f\n",
           isFeasible(ssm_instance) ? "true" : "false", getObjectiveValue(ssm_instance));
    destroyTransportationProblem(ssm_instance);

    // Clean up the base instance.
    destroyTransportationProblem(base_problem);
    
    return 0;
}
