function [U,a,b,Pi,out] = iRBBS4PRW(X,Y,r,c,eta,K,a,b,U,opts)
% An iRBBS method for solving the subproblem in projection robust
% Wassertein (PRW) distance:
%
% min f(U) s.t. U'*U = I,
%
% f(U) = min_{a,b} r'*a + c'*b + eta*log (sum_{ij} K_{ij} exp(-(a_i + b_j + ||U'(x_i - y_j)||^2)/eta))
%
% For a given eta, iRBBS can be also used to compute an approximate
% O(eta)-stationary point of the original PRW distance:
%  
% min_{U,a,b} r'*a + c'*b + max_{ij} {-(a_i + b_j + ||U'(x_i - y_j)||^2)}
% s.t.  U'*U = I; 
% 
% Note that the optimal objective function value of the above problem is
% the opposite of the original PRW distance. 
% Input:
%       (X,Y)    --- data matrices 
%       (r,c)    --- discrete probability measures
%       eta      --- regularization paramter 
%       K        --- estimation of the transportation matrix, usually, we set
%                    it as an all one matrix
%       (a,b,U)  --- initial guess of the variables 
%
%       opts --- option structure with fields:
%            record = 0, no print out 
%            gtol_U    stop control for variable U
%            gtol_ab   stop control for variables a and b
%            maxiter   max number of iterations 
%            BBtype    choices of the BB stepsizes 
%            tau       initial stepsize
%            theta     parameter for inexact condition
%            delta     parameter for linesearch
%            sk_type   use 'log' or 'exp' type of Sinkhorn, if eta is too
%                      small, please choose 'log'
%
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
%% copy some parameters
if ~isfield(opts,'record');  opts.record = 0; end 
if isfield(opts,'gtol_ab') + isfield(opts,'gtol_ab') <= 1
    C0 = (-2*X')*Y + sum(Xt.^2,2) + sum(Y.^2,1);
    C0 = max(C0,0); max_XY = max(C0(:));
    opts.gtol_ab = 1e-6*max(norm(r,'inf'),norm(c,'inf')); 
    opts.gtol_U =  2*max_XY*opts.gtol_ab;  
end 
if ~isfield(opts,'maxiter');  opts.maxiter = 20000; end 
if ~isfield(opts,'BBtype');   opts.BBtype = 'new'; end 
if ~isfield(opts,'tau');      opts.tau = 1e-3; end 
if ~isfield(opts,'theta');      opts.theta = 0.1; end 
if ~isfield(opts,'delta');    opts.delta = 1e-4; end 
if ~isfield(opts,'sk_type');  opts.sk_type = 'exp'; end 

%% copy parameters
gtol_U   = opts.gtol_U;
gtol_ab  = opts.gtol_ab;
maxiter  = opts.maxiter;
record   = opts.record;
delta    = opts.delta;   
BBtype   = opts.BBtype; 
theta      = opts.theta;    
tau      = opts.tau;   
sk_type  = opts.sk_type;
%% define some varibles which will be used later
total_sk_iter = 0;
total_U_iter  = 0;
nfe = 0;
[d,nX] = size(X);   % X: d*n
[~,nY] = size(Y);
C = zeros(nX,nY);
exp_a = exp(-a./eta); exp_b = exp(-b./eta);
Array_sk_info_nfe = zeros(2*maxiter,3);
Array_iter_info   = zeros(maxiter+1,3);
ones_nX = ones(nX,1); ones_nY = ones(nY,1);
max_subiter_sk = 1000;
Kexp_C_exp_b = zeros(nY,1);
Kexp_C = zeros(nX,nY);
logr = log(r); logc = log(c);
Pi = zeros(nX,nY);
scale = gtol_ab/gtol_U; % used in the inexact condition
tol_sub_sk_array = zeros(maxiter+1,1);
tol_sub_sk_array(1,1) = 2;
level = 0.05; % parameter for the new BB stepsizes
%% iter 0
iter = 0;
if K == ones(nX,nY)
    K_tag = 'ones'; % for saving cost when K is an all one matrix
end
%%  compute the initial inexact gradient/function information
nrm_gradU = inf;
tol_sub_sk = max(theta*scale*nrm_gradU,gtol_ab);
if theta < inf % use a more efficient way to compute inexact gradient/function information
    [f,gradU,feasi_ab,iter_sk_sub] = compute_fge_exp(U,tol_sub_sk);
else
    [f,gradU,feasi_ab,iter_sk_sub] = compute_fge_exp_inf(U,tol_sub_sk);
end
nrm_gradU = norm(gradU,'fro');
%% initial setting for nonmonotone linesearch
vs = 0.49*eta;
Ef = f + vs*feasi_ab^2;
fr = Ef; Q  = 1; gamma = 0.85; 
Array_iter_info(1,:) = [nrm_gradU,feasi_ab,iter_sk_sub];
if tau <= 0
    tau = 1/nrm_gradU;
end
nls = 0;
if record >= 1
    str_print = strcat('iter: %2d, nrmgU: %2.1e, nrmgab: %2.1e,',...
        'iter_sk: %4d, step: %2.1e, nls: %4d \n');
    fprintf(1,str_print,iter, nrm_gradU, feasi_ab, iter_sk_sub,tau,nls);
end
d =  - gradU; %./sqrt(rr + 1e-12);
%% starting iterations
for iter = 1:maxiter
    tol_sub_sk_array(iter + 1,1) = feasi_ab; %
    if feasi_ab <= gtol_ab && nrm_gradU <= gtol_U || isnan(nrm_gradU)
        break;
    end
    %% store the information of latest iteration
    Up = U; %gradp = grad; fp = f;
    nrm_gradp = nrm_gradU;
    %% linesearch based on the inexact information
    descent = delta*nrm_gradp^2;
    nls = 0;
    dp = d;
    tol_sub_sk = max(min(theta*scale*nrm_gradU),gtol_ab); %,0.1/iter
    while 1
        U = retr(Up,dp,tau,'polar'); 
        if theta < inf
            [f,gradU,feasi_ab,iter_sk_sub] = compute_fge_exp(U,tol_sub_sk);
        else
            [f,gradU,feasi_ab,iter_sk_sub] = compute_fge_exp_inf(U,tol_sub_sk);
        end
        Ef = f + vs*feasi_ab^2;
        if Ef <= fr - descent - (0.5*eta - vs)*feasi_ab^2  || nls >= 5 % +  0/iter^2
            break;
        end
        nls = nls + 1;
        tau = 0.5*tau;
    end
    %% print information 
    if record >= 1
        str_print = strcat('iter: %2d, nrmgU: %2.1e, nrmgab: %2.1e,',...
            'iter_sk: %4d, step: %2.1e, nls: %4d, f: %14.12e \n');
        fprintf(1,str_print,...
            iter, nrm_gradU, feasi_ab, iter_sk_sub, tau, nls,f);
    end
    Array_iter_info(iter+1,:) = [nrm_gradU,feasi_ab,iter_sk_sub];

    %% compute the BB stepsize
    d = -gradU;
    dgrad = d - dp;
    dU    = U - Up;
    switch BBtype
        case 'LBB'
            tau = (dU(:)'*dU(:)) / abs(dU(:)'*dgrad(:));
        case 'SBB'
            tau =  abs(dU(:)'*dgrad(:)) / (dgrad(:)'*dgrad(:));
        case 'ABB'
            temp4ABB = abs(dU(:)'*dgrad(:));
            if mod(iter,2) == 0
                tau = (dU(:)'*dU(:)) / temp4ABB;
            else
                tau =  temp4ABB / (dgrad(:)'*dgrad(:));
            end
        case 'ADBB'
            temp4ABB = abs(dU(:)'*dgrad(:));
            tau_L = (dU(:)'*dU(:)) / temp4ABB;
            tau_S =  temp4ABB / (dgrad(:)'*dgrad(:));
            tau_ratio = tau_S/tau_L;
            if tau_ratio <= level
                tau = tau_S;
                level = level/1.02;
            else
                tau = tau_L;
                level = level*1.02;
            end
        case 'new'
            % this is a variant of the new efficient BB stepsize proposed
            % by Y. K. Huang, Y.H. Dai, X.W. Liu, Equipping the Barzilai--Borwein 
            % method with the two dimensional quadratic termination
            % property, SIAM Journal on Optimization, 31(4): 3068-3096,
            % 2021. 
            % 
            if iter > 1
                bb1_old = bb1;
                bb2_old = bb2;
            end
            sty = abs(dU(:)'*dgrad(:));
            bb1 = (dU(:)'*dU(:)) / sty;
            bb2 =  sty / (dgrad(:)'*dgrad(:));
            if iter == 1
                bb1_old = bb1;
                bb2_old = bb2;
            end
            if bb2/bb1 <= level
                phi13 =  (bb2_old - bb2)/(bb2_old * bb2 * (bb1_old - bb1));
                phi23 =  phi13 * bb1_old + 1/bb2_old;
                bb_new = 2/( phi23 + sqrt(phi23^2 - 4*phi13));
                tau = min([bb2_old, bb2, max(bb_new,0)]);
                level = level/1.02;
            else
                tau = bb1;
                level = level*1.02;
            end
    end
    tau = max(1e-10,min(tau,1e10));
    Qp = Q; Q = gamma*Qp + 1; fr = (Ef + gamma*Qp*fr)/Q;
end
Array_iter_info = Array_iter_info(1:iter+1,:);
%% do postprocessing
if strcmp(sk_type,'exp')
    Pi = (exp_a.*Kexp_C).*exp_b';
end
Pi_round = Pi_post(Pi,r,c); % to generate a feasible Pi
%% outputs
out.tau = tau;
out.PRW = sum(sum(Pi_round.*C));
Z = a + b' + C;
minZ = min(min(Z));
Z = Z - minZ;
f = r'*a + c'*b - minZ;
%% post processing
grad = compute_partial_grad_U(U,Pi_round);
out.f = f;
out.C = C;
out.a = a;
out.b = b;
out.Pi = Pi;
out.Pi_round = Pi_round;
out.CPL = sum(sum(Pi_round.*Z));
out.nrm_gradU = norm(grad,'fro');
out.feasi_ab  = feasi_ab;
out.iter = iter;
out.nfe = nfe;
out.total_sk_iter = total_sk_iter;
out.total_U_iter = total_U_iter;
out.Array_iter_info = Array_iter_info;
out.Array_sk_info_nfe = Array_sk_info_nfe(1:nfe,:);
if iter == maxiter
    stop_message = 'out of maxiter';
else
    stop_message = 'successful: kkt';
end
out.stop_message = stop_message;

%% nested functions
    function [grad] = compute_partial_grad_U(U,Pi)
        Pi2_temp = Pi*ones_nY;
        UtY_temp = U'*Y;
        XtU_temp = X'*U; YtU_temp = UtY_temp';
        Pi1_temp = Pi'*ones_nX;
        G_temp = X*(Pi2_temp.*XtU_temp - Pi*YtU_temp) + Y*(Pi1_temp.*YtU_temp - Pi'*XtU_temp);
        G_temp = (-2)*G_temp;
        UtG_temp  = U'*G_temp;
        grad = G_temp - U*UtG_temp;
    end

    function [f,grad,feasi_ab,iter_sk_sub] = compute_fge_exp(U,tol_sub_sk)
        %  f(U) = min_{a,b} (r'*a + c'*b + eta* sum_{ij} K_{ij} exp(-(a_i + b_j + ||U'(x_i - y_j)||^2)/eta))
        UtY = U'*Y;
        temp2 = sum(UtY.^2,1);
        XtU = X'*U; YtU = UtY';
        temp1 = sum(XtU.^2,2);
        C = (-2*XtU)*UtY + temp1 + temp2; % C_ij = ||U'(x_i - y_j)||^2
        switch sk_type
            case 'exp'
                if iter <= 0
                    max_subiter_sk = 1;
                else
                    max_subiter_sk = 1000; 
                end
                if strcmp(K_tag,'ones')
                    Kexp_C = exp(-C./eta);
                else
                    Kexp_C = K.*exp(-C./eta);
                end
                exp_b = exp(-b./eta);
                Kexp_C_exp_b = Kexp_C*exp_b;
                for iter_sk_sub = 1: max_subiter_sk
                    %% sinkhorn's iterations
                    exp_a = r./(Kexp_C_exp_b);
                    exp_b = c./(Kexp_C'*exp_a);
                    %% compute the feasiblity violation
                    Kexp_C_exp_b = Kexp_C*exp_b;
                    Pi2 = exp_a.*(Kexp_C_exp_b);
                    % it is faster than computing Pi = (exp_a.*Kexp_C).*exp_b'
                    % first, and then computing the feasibility violation
                    feasi_ab = norm(Pi2 - r,1);
                    if  isnan(sum(exp_a))
                        break;
                    end
                    if feasi_ab <= tol_sub_sk
                        break;
                    end
                end
                a = -eta*log(exp_a); b = -eta*log(exp_b);
                G = X*(Pi2.*XtU - exp_a.*(Kexp_C*(exp_b.*YtU))) + Y*(c.*YtU - exp_b.*(Kexp_C'*(exp_a.*XtU)));
            case 'log'
                if iter <= 0
                    max_subiter_sk = 1;
                else
                    max_subiter_sk = 100; 
                end
                u = a./eta; v = b./eta;
                C_temp = C./eta + u + v';
                logsum_1 =  logsumexp_row(C_temp);
                du = - logr  + logsum_1;
                u = u + du;
                for iter_sk_sub = 1:max_subiter_sk
                    C_temp = C_temp + du;
                    logsum_2 = logsumexp_column(C_temp);
                    dv = - logc  + logsum_2;
                    v = v + dv;
                    C_temp = C_temp + dv';
                    logsum_1 = logsumexp_row(C_temp);
                    Pi2 = exp(logsum_1);
                    feasi_ab = norm(Pi2 - r,1);
                    du = - logr  + logsum_1;
                    u = u + du;
                    if feasi_ab <= tol_sub_sk
                        break;
                    end
                end
                Pi = exp(-C_temp);
                a = eta*u; b = eta*v;
                Pi1 = c;
                Pi2 = Pi*ones_nY;
                G = X*(Pi2.*XtU - Pi*YtU) + Y*(Pi1.*YtU - Pi'*XtU);
        end
        tol_sub_sk = feasi_ab;
        G = (-2)*G;
        UtG  = U'*G;
        grad = G - U*UtG;
        % X: d*n, Pi: n*n, U: d*k, XtU: n*k
        nrm_gradU = norm(grad,'fro');
        total_U_iter  = total_U_iter  + 1;
        nfe = nfe + 1;
        Array_sk_info_nfe(nfe,:) = [feasi_ab, iter_sk_sub, tol_sub_sk];
        ab = proj_affine([a;b]);
        a = ab(1:nX); b = ab(nX+1:nX+nY); % this is very important to avoid
        f = r'*a + c'*b;
        %% global variables
        total_sk_iter = total_sk_iter + iter_sk_sub;
        if record >= 10
            fprintf(1,'\t iter_sk: %4d\n',iter_sk_sub);
        end
    end

    function [f,grad,feasi_ab,iter_sk_sub] = compute_fge_exp_inf(U,tol_sub_sk)
        %  f(U) = min_{a,b} (r'*a + c'*b + eta* sum_{ij} K_{ij} exp(-(a_i + b_j + ||U'(x_i - y_j)||^2)/eta))
        iter_sk_sub = 1;
        UtY = U'*Y;
        temp2 = sum(UtY.^2,1);
        XtU = X'*U; YtU = UtY';
        temp1 = sum(XtU.^2,2);
        C = (-2*XtU)*UtY + temp1 + temp2; % C_ij = ||U'(x_i - y_j)||^2
        switch sk_type
            case 'exp'
                if strcmp(K_tag,'ones')
                    Kexp_C = exp(-C./eta);
                else
                    Kexp_C = K.*exp(-C./eta);
                end
                exp_b = exp(-b./eta);
                exp_a = r./(Kexp_C*exp_b);
                exp_b = c./(Kexp_C'*exp_a);
                Pi2 = exp_a.*(Kexp_C*exp_b);
                feasi_ab = norm(Pi2 - r,1); % max(norm(K2 - r), norm(K1 - c));
                Pi1 = c;
                G = X*(Pi2.*XtU - exp_a.*(Kexp_C*(exp_b.*YtU))) + Y*(Pi1.*YtU - exp_b.*(Kexp_C'*(exp_a.*XtU)));
                a = -eta*log(exp_a); b = -eta*log(exp_b);
            case 'log'
                u = a./eta; v = b./eta;
                C_temp = C./eta + u + v';
                logsum_1 = logsumexp_row(C_temp);
                du = - logr  + logsum_1;
                u = u + du;
                C_temp = C_temp + du;
                logsum_2 = logsumexp_column(C_temp);
                dv = - logc  + logsum_2;
                v = v + dv;
                a = eta*u; b = eta*v;
                C_temp = C_temp + dv';
                Pi = exp(-C_temp);
                Pi2 = Pi*ones_nY;
                feasi_ab = norm(Pi2 - r,1);
                Pi1 = Pi'*ones_nX;
                G = X*(Pi2.*XtU - Pi*YtU) + Y*(Pi1.*YtU - Pi'*XtU);
        end
        G = (-2)*G;
        UtG  = U'*G;
        grad = G - U*UtG;
        % X: d*n, Pi: n*n, U: d*k, XtU: n*k
        nrm_gradU = norm(grad,'fro');
        total_U_iter  = total_U_iter  + 1;
        nfe = nfe + 1;
        Array_sk_info_nfe(nfe,:) = [feasi_ab, iter_sk_sub, tol_sub_sk];
        ab = proj_affine([a;b]);
        a = ab(1:nX); b = ab(nX+1:nX+nY); % this is very important to avoid
        f = r'*a + c'*b;
        %% global variables
        total_sk_iter = total_sk_iter + iter_sk_sub;
        if record >= 10
            fprintf(1,'\t iter_sk: %4d\n',iter_sk_sub);
        end
    end

    function Z = retr(U,D,tau,retr_type)
        Z = U + tau*D;
        switch retr_type
            case 'polar'
                Z = Z*(Z'*Z)^(-0.5);
            case 'qr'
                LL = chol(Z'*Z);
                Z = Z/LL;
        end
    end

    function y = proj_affine(x)
        % projection onto [r' -c']x = 0
        aa = x(1:nX); bb = x(nX+1:nX+nY);
        temp = (c'*bb - r'*aa)/2;
        y = [aa + temp; bb - temp];
    end


    function a = logsumexp_row(A)
        %  logsum_j exp(-A_{j}) = logsum_j exp( -(A_{j} - A_min)) - A_min
        Amin = min(A,[],2);
        A = Amin - A;
        temp = exp(A);
        a = log(sum(temp,2)) - Amin;
    end

    function a = logsumexp_column(A)
        Amin = min(A,[],1);
        A = Amin - A;
        temp = exp(A);
        a = log(sum(temp,1)) - Amin;
        a = a';
    end

end