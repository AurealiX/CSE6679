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
#include "gpu_lcm.h"
#include "cpu_ssm.h"
#include "gpu_vam.h"
#include "gpu_ssm.h"

// Clone instance helper
TransportationProblem* cloneTransportationProblem(TransportationProblem* original) {
    int m = original->numSupply, n = original->numDemand;
    TransportationProblem* copy = createTransportationProblem(m, n);
    memcpy(copy->supply, original->supply, m*sizeof(double));
    memcpy(copy->demand, original->demand, n*sizeof(double));
    for (int i = 0; i < m; i++) {
        memcpy(copy->cost[i], original->cost[i], n*sizeof(double));
        memcpy(copy->solution[i], original->solution[i], n*sizeof(double));
        memcpy(copy->BFS[i], original->BFS[i], n*sizeof(int));
    }
    return copy;
}

// Helper function to solve and report results
double solveAndReport(TransportationProblem* inst, const char* method, double (*solveFunc)(TransportationProblem*)) {
    double time = solveFunc(inst);
    printf("%s: feas=%s, obj=%f\n", 
           method, 
           isFeasible(inst) ? "true" : "false", 
           getObjectiveValue(inst));
    return time;
}

int main() {
    int num_supply = 100, num_demand = 57, max_cost = 10, seed = 1;
    printf("Enter seed: ");
    scanf("%d", &seed);
    printf("Enter supply nodes: ");
    scanf("%d", &num_supply);
    printf("Enter demand nodes: ");
    scanf("%d", &num_demand);
    int max_supply = num_supply + num_demand;
    
    // New: 4-digit binary mode (VAM, LCM, MODI, SSM)
    char mode[5];
    printf("Enter 4-digit binary mode (VAM LCM MODI SSM): ");
    scanf("%4s", mode);
    
    double vamTime = 0, lcmTime = 0, modiTime = 0, ssmTime = 0;
    TransportationProblem* inst;
    
    // Phase 1: VAM test
    inst = createTransportationProblem(num_supply, num_demand);
    generateRandomInstance(inst, num_supply, num_demand, max_supply, 0, max_cost, seed);
    if (mode[0] == '0') {
        vamTime = solveAndReport(inst, "CPU VAM", vamCPUSolve);
    } else {
        vamTime = solveAndReport(inst, "GPU VAM", gpuVamSolve);
    }
    destroyTransportationProblem(inst);
    
    // Phase 1: LCM test
    inst = createTransportationProblem(num_supply, num_demand);
    generateRandomInstance(inst, num_supply, num_demand, max_supply, 0, max_cost, seed);
    if (mode[1] == '0') {
        lcmTime = solveAndReport(inst, "CPU LCM", lcmCPUSolve);
    } else {
        lcmTime = solveAndReport(inst, "GPU LCM", lcmGPUSolve);
    }
    destroyTransportationProblem(inst);
    
    // Phase 2: Use one base instance for MODI and SSM.
    TransportationProblem* base_problem = createTransportationProblem(num_supply, num_demand);
    generateRandomInstance(base_problem, num_supply, num_demand, max_supply, 0, max_cost, seed);
    // Base BFS computed using CPU LCM (unchanged)
    lcmCPUSolve(base_problem);
    
    // Phase 2: MODI test
    inst = cloneTransportationProblem(base_problem);
    if (mode[2] == '0') {
        modiTime = solveAndReport(inst, "CPU MODI", modiCPUSolve);
    } else {
        modiTime = solveAndReport(inst, "GPU MODI", modiGPUSolve);
    }
    destroyTransportationProblem(inst);
    
    // Phase 2: SSM test
    inst = cloneTransportationProblem(base_problem);
    if (mode[3] == '0') {
        ssmTime = solveAndReport(inst, "CPU SSM", ssmCPUSolve);
    } else {
        ssmTime = solveAndReport(inst, "GPU SSM", ssmGPUSolve);
    }
    destroyTransportationProblem(inst);
    
    destroyTransportationProblem(base_problem);
    
    // Final times
    printf("Times:\nVAM: %f sec\nLCM: %f sec\n", vamTime, lcmTime);
    printf("MODI (%s): %f sec\n", mode[2] == '0' ? "CPU" : "GPU", modiTime);
    printf("SSM (%s): %f sec\n", mode[3] == '0' ? "CPU" : "GPU", ssmTime);
    
    return 0;
}