# ReALM and iRBBS
Codes for our paper:

Bo Jiang, Ya-Feng Liu, A Riemannian Exponential Augmented Lagrangian Method for Computing the Projection Robust Wasserstein Distance, 
accepted by NeurIPS 2023. 

# Authors

We hope the codes are useful for your research and application. If you have any bug reports or comments, please kindly email one of the authors 

- Bo Jiang, jiangbo@njnu.edu.cn
- Ya-Feng Liu, yafliu@lsec.cc.ac.cn

# Data generation: 

We generated the Shakespeare and Mnist data using the approach outlined in RAGAS package 
(Python, https://github.com/fanchenyou/PRW) and saved the corresponding matfiles. 

# Codes description 

1. To generate Tables 1 and 3 in our paper,
   - Run demo_PRW_fixed_eta_for_shakespeare_mnist.m with nRun = 20;
   - Run Tables1_3_shakespeare_mnist_iRBBS_RABCD.m (please modify some settings following the remarks in PRW_fixed_eta_for_shakespeare_mnist.m)

2. To generate Tables 2 and 5 in our paper, 
   - demo_PRW_compare_ReALM_for_shakespeare_mnist.m with nRun = 20; 
   - Run Tables2_5_shakespeare_mnist_REALM (please modify some settings following the remarks in PRW_compare_ReALM_for_shakespeare_mnist.m)

3. Some description on the remaing codes: 
   - iRBBS4PRW.m is the main code for iRBBS in our paper;
   - iRBBS_for_ReALM.m is nearly identical to iRBB4PWR.m but has been rewritten for easier integration in ReALM4PRW.m
   - ReALM4PRW.m is the main code for REALM in our paper, and it will invoke  iRBBS_for_ReALM.m;
   - RABCD4PRW.m and RBCD4PRW.m are the codes of R(A)BCD proposed by Huang, Ma \& Lai (2021);
     For the authors' implementation version of R(A)BCD, please visit: https://github.com/mhhuang95/PRW_RBCD 
   - Pi_post.m is to round a matrix as a feasible matrix in Pi(r,c); 
   - PRW_fixedU_Mosek.m: after obtain a nearly optimal U, we solve a standard optimal transport problem by Mosek; 
   - PRW_fixedU_Gurobi.m: after obtain a nearly optimal U, we solve a standard optimal transport problem by Gurobi;
   - linprog_gurobi.m is invoked by PRW_fixedU_Gurobi.m;
   - generate_initial_U.m generated an initial point of U by solving an eigenvaule problem;
   - compute_C.m computes the matrix $C$ with $C_{ij} = \|U^T(x_i - y_j)\|^2$;

A kind reminder: To run the codes, please ensure you have Mosek and Gurobi installed. If not, you can comment out the corresponding scripts.

# References: 

Our work is based on RAGAS and RABCD. For more details on these two methods, please refer to the following references: 

1. T. Lin, C. Fan, N. Ho, M. Cuturi, and M. Jordan, Projection robust Wasserstein distance and Riemannian
  optimization, Advances in Neural Information Processing Systems. 33: 9383-9397, 2020.

2. M. Huang, S. Ma, and L. Lai, Projection robust Wasserstein barycenters, International Conference on Machine Learning, pages
  4456-4465. PMLR, 2021. 


