function [Pi,U,a,b,out] = RBCD4PRW(X,Y,r,c,eta,K,b,U,opts)
% A Riemannian block coordinate descent method (RBCD) for computing an
% O(eta)-stationary point of the original projection robust
% Wassertein (PRW) distance problem.
% This method solves the following regularization problem
% min f(U,a,b) s.t. U'*U = I,
% where  f(U,a,b) =  r'*a + c'*b
%                    + eta* log (sum_{ij} K_{ij} exp(-(a_i + b_j + ||U'(x_i - y_j)||^2)/eta))
%  For more details on RBCD, one can refer to
%
%  M. Huang, S. Ma, L. Lai, A Riemannian block coordinate descent method
%   for computing the projection robust Wasserstein distance, ICML 2021.
%
% Remark:
%   Since the authors provided their code in Python, we implemented this method ourselves.
%
%   Bo Jiang, Ya-Feng Liu.
%

%% copy some parameters
gtol_U   = opts.gtol_U;
gtol_ab  = opts.gtol_ab;
maxiter  = opts.maxiter;
record   = opts.record;
tau = opts.tau;
[~,nX] = size(X);
[~,nY] = size(Y);
C = zeros(nX,nY);
exp_b = exp(-b./eta);
Array_iter_info  = zeros(maxiter+1,2);
if K == ones(nX,nY)
    K_tag = 'ones';
end
%% starting iterations
for iter = 1:maxiter
    %% update a,b,U
    UtX = U'*X; UtY = U'*Y;
    temp1 = sum(UtX.^2,1)'; temp2 = sum(UtY.^2,1);
    XtU = UtX'; YtU = UtY';
    C = (-2*UtX')*UtY + temp1 + temp2; % C_ij = ||U'(x_i - y_j)||^2
    %     C = max(C,0);
    if strcmp(K_tag,'ones')
        Kexp_C = exp(-C./eta);
    else
        Kexp_C = K.*exp(-C./eta);
    end
    % one sinkhorn iteration
    exp_a = r./(Kexp_C*exp_b);
    exp_b = c./(Kexp_C'*exp_a);
    Pi2 = exp_a.*(Kexp_C*exp_b); % Pi*ones_nY;
    Pi1 = c; %Pi'*ones_nX;
    feasi_ab = norm(Pi2 - r,1); % max(norm(K2 - r), norm(K1 - c));
    %% update U
    G = X*(Pi2.*XtU - exp_a.*(Kexp_C*(exp_b.*YtU))) + Y*(Pi1.*YtU - exp_b.*(Kexp_C'*(exp_a.*XtU)));
    G = (-2)*G;
    UtG  = U'*G;
    grad = G - U*UtG;
    % X: d*n, Pi: n*n, U: d*k, XtU: n*k
    nrm_grad = norm(grad,'fro');
    a = -eta*log(exp_a); b = -eta*log(exp_b);
    ab = proj_affine([a;b]);
    a = ab(1:nX); b = ab(nX+1:nX+nY); % this is very important to avoid
    exp_b = exp(-b./eta);

    Array_iter_info(iter,:) = [nrm_grad,feasi_ab];
    if feasi_ab <= gtol_ab && nrm_grad <= gtol_U
        break;
    end
    U = retr(U,-grad,tau,'polar');
    if record >= 1
        a = -eta*log(exp_a);
        b = -eta*log(exp_b);
        f = r'*a + c'*b;
        str_print = strcat('iter: %2d, nrmgU: %2.1e, nrmgab: %2.1e,',...
            'step: %2.1e, f: %14.12e\n');
        fprintf(1,str_print,...
            iter, nrm_grad, feasi_ab, tau, f);
    end
end
Pi = (exp_a.*Kexp_C).*exp_b';

a = -eta*log(exp_a);
b = -eta*log(exp_b);
Z = a + b' + C;
minZ = min(min(Z));
Z = Z - minZ;
f = r'*a + c'*b - minZ;

Array_iter_info = Array_iter_info(1:iter,:);
%% do postprocessing
Pi =  Pi_post(Pi,r,c);
%% outputs
out.PRW = sum(sum(Pi.*C));
out.f = f;
out.C = C;
out.a = a;
out.b = b;
out.Pi = Pi;
out.nrm_grad = nrm_grad;
out.feasi_ab  = feasi_ab;
out.iter = iter;
out.Array_iter_info = Array_iter_info;
if iter == maxiter
    stop_message = 'out of maxiter';
else
    stop_message = 'successful: kkt';
end
out.stop_message = stop_message;
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

end