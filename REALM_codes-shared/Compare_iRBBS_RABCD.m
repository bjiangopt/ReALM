function [Perf_iRBBS,Perf_RBCD,Perf_RABCD] =  Compare_iRBBS_RABCD(X,Y,r,c,k,eta,seed,opts)
%% 
doiRBBS  = opts.doiRBBS;
doRBCD  = opts.doRBCD;
doRABCD = opts.doRABCD;
doPost  = opts.doPost;
data_type = opts.data_type;
if strcmp(data_type,'shakespeare')  
    art1 = opts.art1; 
    art2 = opts.art2;
else 
    art1 = ''; % for synthetic datasets
    art2 = '';
end

if doiRBBS;  theta_list = opts.theta_list;   nan_iRBBS = 0;  end
if doRBCD;   tau_RBCD = opts.tau_RBCD;   nan_RBCD = 0;  end
if doRABCD;  tau_RABCD = opts.tau_RABCD; nan_RABCD = 0; end

%% settings 
Xt = X';
C0 = (-2*X')*Y + sum(Xt.^2,2) + sum(Y.^2,1);
C0 = max(C0,0);
max_XY = max(C0(:)); % for scaled Riemannian gradient tolerance
gtol_ab = 1e-6*max(norm(r,'inf'),norm(c,'inf'));
gtol_U =  2*max_XY*gtol_ab;  
clear opts
%% generate starting points
nY = size(Y,2);
nX = size(X,2);
a0 = zeros(nX,1);
b0 = zeros(nY,1);
U0 = generate_initial_U(seed,X,Y,r,c);
Pi0 = ones(nX,nY);
%% outputs 
Perf_iRBBS = zeros(14,1);
Perf_RBCD  = zeros(13,1);
Perf_RABCD = zeros(13,1);
if doiRBBS
    n_theta = length(theta_list);
    Perf_iRBBS  = zeros(14,n_theta);
    for i_theta = 1:n_theta
        clear opts
        eta_iRBBS = eta;
        theta = theta_list(i_theta);
        %% iRBBS
        opts.gtol_U  = gtol_U;
        opts.gtol_ab = gtol_ab;
        opts.maxiter = 20000;
        opts.record  = 0;
        opts.delta   = 1e-4;
        opts.BBtype  = 'new';
        opts.theta = theta;
        opts.tau = 1e-3;
        opts.dopost = 1;
        opts.sk_type = 'exp'; 
        t0 = tic;
        [U_iRBBS,a_iRBBS,b_iRBBS,Pi_iRBBS,out_iRBBS] = ...
            iRBBS4PRW(X,Y,r,c,eta_iRBBS,Pi0,a0,b0,U0,opts);
        t_iRBBS = toc(t0);
        f_iRBBS = out_iRBBS.f;
        PRW_iRBBS = out_iRBBS.PRW;
        nrm_grad_iRBBS = out_iRBBS.nrm_gradU;
        feasi_ab_iRBBS  = out_iRBBS.feasi_ab;
        iter_iRBBS = out_iRBBS.iter;
        nfe_iRBBS = out_iRBBS.nfe;
        sk_iter_iRBBS = out_iRBBS.total_sk_iter;
        %% post
        if ~sum(isnan(U_iRBBS))
            t0 = tic;
            if doPost
                if strcmp(art1,'TM') && strcmp(art2,'T') || strcmp(art1,'I') && strcmp(art2,'T') %|| strcmp(data_type,'mnist')
                    [Pi_iRBBS_post,a_iRBBS_post,b_iRBBS_post] = PRW_fixedU_Mosek(U_iRBBS,X,Y,r,c);
                else
                    [Pi_iRBBS_post,a_iRBBS_post,b_iRBBS_post] = PRW_fixedU_Gurobi(U_iRBBS,X,Y,r,c,0);
                end
            else
                Pi_iRBBS_post = Pi_iRBBS;
                a_iRBBS_post = a_iRBBS;
                b_iRBBS_post = b_iRBBS;
            end
            t_iRBBS_post = toc(t0);
            C_iRBBS = compute_C(U_iRBBS,X,Y);
            PRW_iRBBS_post = sum(sum(C_iRBBS.*Pi_iRBBS_post));
            Z_post = a_iRBBS_post + b_iRBBS_post' + C_iRBBS;
            f_iRBBS_post = r'*a_iRBBS_post + c'*b_iRBBS_post; 
        else
            nan_iRBBS = 1;
            t_iRBBS_post = -1;
            PRW_iRBBS = -1;
            PRW_iRBBS_post = -1;
            f_iRBBS_post = -1;
        end
        Perf_iRBBS(:,i_theta) = ...
            [theta,eta_iRBBS, nrm_grad_iRBBS,feasi_ab_iRBBS,...
            iter_iRBBS,nfe_iRBBS, sk_iter_iRBBS, t_iRBBS,t_iRBBS_post, ...
            PRW_iRBBS, PRW_iRBBS_post, f_iRBBS, f_iRBBS_post,nan_iRBBS]';
    end
end

%% RBCD
if doRBCD
    opts.gtol_U  = gtol_U;
    opts.gtol_ab = gtol_ab;
    opts.maxiter = 5000;20000;
    opts.record  = 0;
    opts.tau = tau_RBCD/eta;
    opts.dopost  = 1;
    t0 = tic;
    [Pi_RBCD,U_RBCD,a_RBCD,b_RBCD,out_RBCD] = RBCD4PRW(X,Y,r,c,eta,Pi0,b0,U0,opts);
    t_RBCD = toc(t0);
    f_RBCD = out_RBCD.f;
    PRW_RBCD = out_RBCD.PRW;
    nrm_grad_RBCD = out_RBCD.nrm_grad;
    feasi_ab_RBCD = out_RBCD.feasi_ab;
    iter_RBCD = out_RBCD.iter;
    nfe_RBCD  = out_RBCD.iter;
    sk_iter_RBCD = out_RBCD.iter;
    %% post
    if ~sum(isnan(U_RBCD))
        t0 = tic;
        if doPost
            if strcmp(art1,'TM') && strcmp(art2,'T')  || strcmp(art1,'I') && strcmp(art2,'T')
                [Pi_RBCD_post,a_RBCD_post,b_RBCD_post] = PRW_fixedU_Mosek(U_RBCD,X,Y,r,c);
            else
                [Pi_RBCD_post,a_RBCD_post,b_RBCD_post] = PRW_fixedU_Gurobi(U_RBCD,X,Y,r,c,0);
            end
        else
            Pi_RBCD_post = Pi_RBCD;
            a_RBCD_post  = a_RBCD;
            b_RBCD_post  = b_RBCD;
        end
        t_RBCD_post = toc(t0);
        C_RBCD = compute_C(U_RBCD,X,Y);
        PRW_RBCD_post = sum(sum(C_RBCD.*Pi_RBCD_post));
        f_RBCD_post = r'*a_RBCD_post + c'*b_RBCD_post; % + eta*sum(sum(PRW_RBCD_post));
    else
        nan_RBCD =  1;
        t_RBCD_post = -1;
        PRW_RBCD = -1;
        PRW_RBCD_post = -1;
        f_RBCD_post = -1;
    end
    Perf_RBCD = ...
        [eta, nrm_grad_RBCD,feasi_ab_RBCD,...
        iter_RBCD,nfe_RBCD, sk_iter_RBCD, t_RBCD,...
        t_RBCD_post, PRW_RBCD, PRW_RBCD_post, f_RBCD, f_RBCD_post,nan_RBCD]';
end
%% RABCD
if doRABCD
    opts.gtol_U  = gtol_U;
    opts.gtol_ab = gtol_ab;
    opts.maxiter = 20000;
    opts.record  = 0;
    opts.tau =   tau_RABCD/eta;
    opts.dopost  = 1;
    t0 = tic;
    [Pi_RABCD,U_RABCD,a_RABCD,b_RABCD,out_RABCD] = RABCD4PRW(X,Y,r,c,eta,Pi0,b0,U0,opts);
    t_RABCD = toc(t0);
    f_RABCD = out_RABCD.f;
    PRW_RABCD = out_RABCD.PRW;
    nrm_grad_RABCD = out_RABCD.nrm_grad;
    feasi_ab_RABCD  = out_RABCD.feasi_ab;
    iter_RABCD = out_RABCD.iter;
    nfe_RABCD = out_RABCD.iter;
    sk_iter_RABCD = out_RABCD.iter;
    %% post
    if ~sum(isnan(U_RABCD))
        t0 = tic;
        if doPost
            if strcmp(art1,'TM') && strcmp(art2,'T')  || strcmp(art1,'I') && strcmp(art2,'T')
                [Pi_RABCD_post,a_RABCD_post,b_RABCD_post] = PRW_fixedU_Mosek(U_RABCD,X,Y,r,c);
            else
                [Pi_RABCD_post,a_RABCD_post,b_RABCD_post] = PRW_fixedU_Gurobi(U_RABCD,X,Y,r,c,0);
            end
        else
            Pi_RABCD_post = Pi_RABCD;
            a_RABCD_post  = a_RABCD;
            b_RABCD_post  = b_RABCD;
        end
        t_RABCD_post = toc(t0);
        C_RABCD = compute_C(U_RABCD,X,Y);
        PRW_RABCD_post = sum(sum(C_RABCD.*Pi_RABCD_post));
        f_RABCD_post = r'*a_RABCD_post + c'*b_RABCD_post; % + eta*sum(sum(PRW_RABCD_post));
    else
        nan_RABCD = 1;
        t_RABCD_post = -1;
        PRW_RABCD = -1;
        PRW_RABCD_post = -1;
        f_RABCD_post = -1;
    end
    Perf_RABCD = ...
        [eta, nrm_grad_RABCD,feasi_ab_RABCD,...
        iter_RABCD,nfe_RABCD, sk_iter_RABCD, t_RABCD,...
        t_RABCD_post, PRW_RABCD, PRW_RABCD_post, f_RABCD, f_RABCD_post,nan_RABCD]';
end

%% nested function to generate the initial point of U
    function U0 = generate_initial_U(seed,X,Y,r,c)
        rng(seed);
        K  = rand(nX,nY);
        K  = max(0,K);
        K  = K./sum(sum(K));
        [K,~]  = Pi_post(K,r,c);
        K2 = sum(K,2);
        K1 = sum(K,1)';
        MM = X*(K2.*X' - K*Y') + Y*(K1.*Y' - K'*X');
        [V,~] = eigs(MM,k,'largestreal');
        U0 = V;
    end
end
