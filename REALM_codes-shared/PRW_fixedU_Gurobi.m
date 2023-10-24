function [Pi,a_post,b_post,f,nrm_gradU,lambda,feasi_Pi] = PRW_fixedU_Gurobi(U,X,Y,r,c,method)
%% Given U, the PRW distance problem reduces to a standard optimal transport problem. 
%  Here, we use Gurobi to compute the reduced optimal transport accurately.
%  
C = compute_C(U,X,Y); % C:nr*nc
nr = size(r,1);
nc = size(c,1); % C:nr*nc
e_nr = ones(nr,1);
e_nc = ones(nc,1);
Aeq =  [kron(e_nc',speye(nr)); kron(speye(nc), e_nr')];
beq = [r; c];
l = zeros(nr*nc,1);
u = inf*ones(nr*nc,1);
% options.Display =  'on';
% options.Diagnostics = 'off';
% options.Display =  'off';
% options.Simplex = 'Primal';

% options = mskoptimset('');
% options = mskoptimset(options,'Display','off');
% options = mskoptimset(options,'Diagnostics','off');
% options = mskoptimset(options,'Simplex','off');
% options = mskoptimset(options,'MSK_DPAR_BASIS_TOL_X',1e-9)
% t0 = tic;
% [x,fval,exitflag,output,lambda]=linprog(C(:),[],[],Aeq,beq,l,u,options);
% time_mosek = toc(t0)
options.method = method; 
t0 = tic;
[x,fval,exitflag,output,lambda]=linprog_gurobi(C(:),[],[],Aeq,beq,l,u,options);
t_primal = toc(t0);

if ~isempty(lambda)
    z = lambda.eqlin;
    a_post = z(1:nr);
    b_post = z(nr+1:nr+nc);
    Pi = reshape(x,nr,nc);
    Pi = Pi_post(Pi,r,c);
    feasi_Pi = norm(Pi*e_nc - r,1) + norm(Pi'*e_nr - c,1);
    f = sum(sum(C.*Pi));
    gradU = compute_gradU(Pi,U);
    nrm_gradU = norm(gradU,'fro');
else
    Pi = [];
    a_post = [];
    b_post = [];
    f = [];
    nrm_gradU = [];
    lambda = [];
end

    function grad = compute_gradU(K,U)
    K2 = sum(K,2);
    K1 = sum(K,1)';
    % V2 = X*diag(K2)*X' + Y*diag(K1)*Y' - X*K*Y' - Y*K'*X';
    % V2: d*d, U: d*k
    % G = (-2)*(V2*U);
    XtU = X'*U; YtU = Y'*U;
    G = X*(K2.*XtU - K*YtU) + Y*(K1.*YtU - K'*XtU);
    G = (-2)*G;
    UtG = U'*G;
    grad = G - U*UtG;
    end


end