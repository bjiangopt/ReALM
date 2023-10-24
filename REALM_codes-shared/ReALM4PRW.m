function [Pi,U,a,b,out] = ReALM4PRW(X,Y,r,c,Pi,a,b,U,eta,opts)
%% A Riemannian Exponential Augmented Laagrangian method for solving projection robust Wassertein (PRW) distance
%  
% min_{U,a,b} r'*a + c'*b + max_{ij} {-(a_i + b_j + ||U'(x_i - y_j)||^2)}
% s.t.  U'*U = I; 
% 
% Note that the optimal objective function value of the above problem is
% the opposite of the original PRW distance. 
%
% We use iRBBS for solving the subproblem. The corresponding code is
% 'iRBBS_for_REALM'; 
% 
% Input:
%       (X,Y)    --- data matrices 
%       (r,c)    --- discrete probability measures
%       Pi       --- the initial estimation of the multipler matrix, usually, we set
%                    it as an all one matrix
%       (a,b,U)  --- initial guess of the variables 
%       eta      --- the initial penalty paramter 
%
%       opts --- option structure with fields:
%            record = 0, no print out 
%            maxiter: maximum number of the outer iteration 
%            eta_min: the minimum eta allowed 
%            theta_exp: parameter for inexact condition for exp-type Sinkhorn 
%            theta_log: parameter for inexact condition for log-type Sinkhorn 
%            gtol_U_sub,gtol_ab_sub:    tolerance for the first iteration
%            tol_CPL,gtol_U,gtol_ab:    stopping tolerance  for the outer iteration 
%            ratio_U,ratio_ab,ratio_eta,ratio_CPL:  decrease factors for U, ab and eta
%            maxSubiter,record_sub:     parameters for inner solvers
% Output:
%       Pi        ---  the transportation matrix determined by (U,a,b)
%       (U,a,b)   ---  solution
%       out       ---  output information
%
%
% Reference:
%  Bo Jiang, Ya-Feng Liu, A Riemannian exponential augmented Lagrangian method
%  for computing the projection robust Wasserstein distance, Accepted by NeurIPS 2023.
%
% Author: Bo Jiang, Ya-Feng Liu,
%   Version 0.0 ... 2022/08
%   Version 1.0 ... 2023/10
%
%  Contact information: jiangbo@njnu.edu.cn, yafliu@lsec.cc.ac.cn
n_exp = 0;
n_log = 0;
[d,nX] = size(X);   % X: d*nX
[~,nY] = size(Y);
[~,k] = size(U);
%% copy some parameters
%% tolerance for the first iteration
gtol_U_sub  = opts.gtol_U_sub;
gtol_ab_sub = opts.gtol_ab_sub;
%% stopping tolerance  
tol_CPL  = opts.tol_CPL; % tolerance for CPLmentarity
gtol_U   = opts.gtol_U;
gtol_ab  = opts.gtol_ab;
%% decrease factors for U, ab and eta
ratio_U   = opts.ratio_U;
ratio_ab  = opts.ratio_ab;
ratio_eta = opts.ratio_eta;
ratio_CPL = opts.ratio_CPL;
%% parameters for inner solvers
maxSubiter = opts.maxSubiter;
record_sub = opts.record_sub;
% max_subiter_sk = opts.max_subiter_sk;
%% other parameters
maxiter = opts.maxiter;
record  = opts.record;
eta_min = opts.eta_min;
theta_exp = opts.theta_exp;
theta_log = opts.theta_log;
if record
    str_temp = ['%4s', '%15s', '%15s', '%15s','%15s', '%15s',  '%15s',  '%15s', '%15s', '%15s', '%15s', '%15s', '%15s' '\n'];
    str_head = sprintf(str_temp,'iter', 'eta', '||feasi_ab||',...
        '||grad_U||', '||CPL||','||CPL_real||', 'diff_U', 'tol_ab', 'tol_U', 'iter_sub', 'iter_SK_sub', 'update_Pi', 'f');
    str_num = ['%4d  ','%6.2e  ','   %6.2e  ','   %6.2e  ','    %6.2e','    %6.2e', '    %6.2e',  '   %6.2e',...
        '   %6.2e  ', '    %4d  ', '        %4d',    '   \t\t%d    %8.6e\n'];
    fprintf(1,'=========================  ReALM_PRW solver started =========================\n');
    fprintf(1,'tol_ab: %4.2e, tol_U: %4.2e, tol_CPL: %4.2e\n',gtol_ab,gtol_U,tol_CPL);
    fprintf(1,'%s',str_head);
end
%% some initial settings
total_subiter = 0;
total_iter_sk = 0;
total_nfe = 0;
C = compute_C(U,X,Y);
max_C = max(C(:));
Z = a + b' + C;
W = min(eta*Pi,Z);
CPL = norm(W,'fro');
Z = Z - min(min(Z));
CPL_real = sum(sum(Pi.*Z));
CPL0 = CPL;
if record
    fprintf(1, 'CPL0: %2.1e\n',CPL0);
end 
update_Pi = 0;
num_update_Pi = 0;
fail = 0;
%% set the parameters of the subsolver
opts_sub.maxiter = maxSubiter;
opts_sub.record  = record_sub;
opts_sub.delta   = 1e-4;
opts_sub.BBtype  = 'new';'ABB';
opts_sub.dopost  = 0;
opts_sub.tau     = 1e-3;
total_update_Pi = 0;
if max_C/eta <= 500
    opts_sub.theta = 0.4;   
else
    opts_sub.theta = 10;
end
opts_sub.etap = eta;
ap = a; bp = b;
for iter = 1:maxiter
    %% warm-start: initial points for the next iteration
    app = ap; bpp = bp;
    Up = U; ap = a; bp = b;
    Pi_p = Pi;
    if iter >=2 && out_sub.NLS_fail == 10
        ap = app;
        bp = bpp;
    end
    CPL_p = CPL;
    %% setting the parameters of the subsolver 
    if eta <= 1.001*eta_min
        gtol_U_sub = gtol_U;
        gtol_ab_sub = gtol_ab;
        opts_sub.maxiter =  5000; 
    elseif eta <= 1.2*eta_min  
        opts_sub.maxiter = 5000; 
    else  
        opts_sub.maxiter = 150; 
    end
    opts_sub.gtol_U  = gtol_U_sub;
    opts_sub.gtol_ab = gtol_ab_sub;
    opts_sub.scale_tol = 0;
    opts_sub.theta_log = theta_log;
    opts_sub.theta_exp = theta_exp;
    %% solve the ALM subproblem
    if iter > 1
        opts_sub.tau = 1e-3;  
    end
    if eta == eta_min
    end
    %% subsolvers 
    [Pi,U,a,b,out_sub] = iRBBS_for_ReALM(X,Y,r,c,Pi_p,ap,bp,Up,eta,opts_sub);
    %% summarize the output information after solving the subproblem 
    n_exp = n_exp + out_sub.n_exp;
    n_log = n_log + out_sub.n_log;
    opts_sub.etap = eta;
    iter_sub = out_sub.iter;
    iter_sk_sub = out_sub.total_sk_iter;
    total_iter_sk = total_iter_sk + iter_sk_sub;
    total_subiter  = total_subiter + iter_sub;
    total_nfe = total_nfe + out_sub.nfe;
    %% update the tolerance: CPL, nrm_gradU, feasi_ab
    C = compute_C(U,X,Y);
    Z = a + b' + C;
    W = min(eta*Pi,Z);
    CPL = norm(W,'fro');
    f = r'*a + c'*b + max(max(-Z));
    Z = Z - min(min(Z));
    CPL_real = norm(Pi.*Z,'inf'); 
    nrm_gradU_iter = out_sub.nrm_gradU;
    feasi_ab_iter  = out_sub.feasi_ab;
    diff_U = sqrt(abs(2*k - 2*norm(U'*Up,'fro')^2));  
    %% print information 
    if record
        fprintf(1,str_num,...
            iter,eta,feasi_ab_iter,nrm_gradU_iter,CPL,CPL_real,...
            diff_U,gtol_ab_sub,gtol_U_sub,iter_sub,iter_sk_sub,update_Pi,f);
    end

    if  feasi_ab_iter <= gtol_ab && nrm_gradU_iter <= gtol_U  && (CPL_real <= tol_CPL || eta <= eta_min)%|| CPL <= 1e-4 %...
        stop = 'success';
        break;
    end
    if CPL > CPL_p
        fail = fail + 1;
    end
    %% update eta, gtol_U_sub, and gtol_ab_sub 
    if  total_update_Pi <= opts.max_update_Pi && CPL <= ratio_CPL*CPL_p && min(min(log(Pi))) > -400 ...
        total_update_Pi = total_update_Pi + 1;
        num_update_Pi = num_update_Pi + 1;
    else
        Pi = Pi_p;
        eta = max(ratio_eta*eta, eta_min);
    end
    gtol_U_sub  = max(ratio_U*gtol_U_sub, gtol_U);
    gtol_ab_sub = max(ratio_U*gtol_ab_sub, gtol_ab);
end

if iter == maxiter
    stop = 'maximum iteration';
end
%% rounding of Pi
out.eta = eta;
C = compute_C(U,X,Y);
Z = a + b' + C;
out.f = f; %r'*a + c'*b + max(max(-Z));
Pi = Pi_post(Pi,r,c);
out.PRW = sum(sum(Pi.*C)); out_sub.PRW;
out.stop_message = stop;
out.CPL = CPL;
out.nrm_gradU = nrm_gradU_iter;
out.feasi_ab = feasi_ab_iter;
out.Pi = Pi;
out.iter = iter;
out.total_iter_sk = total_iter_sk;
out.total_subiter = total_subiter;
out.total_nfe = total_nfe;
out.num_update_Pi = num_update_Pi;
out.eta = eta;
out.n_exp = n_exp;
out.n_log = n_log;
end
