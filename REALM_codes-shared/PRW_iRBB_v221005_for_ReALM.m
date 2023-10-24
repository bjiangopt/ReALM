function [Pi,U,a,b,out] = iRBB_for_ReALM(X,Y,r,c,K,a,b,U,eta,opts)

%
%% copy some parameters
gtol_U   = opts.gtol_U;
gtol_ab  = opts.gtol_ab;
maxiter  = opts.maxiter;
record   = opts.record;
delta    = opts.delta;   % parameter for linesearch
BBtype   = opts.BBtype;  % identify which kind of BB stepsize
rho      = opts.rho;     % parameters for inexact condition
tau      = opts.tau;     % initial stepsize
%% initial settings
total_sk_iter = 0;
total_U_iter  = 0;
nfe = 0;
[d,nX] = size(X);   % X: d*n
[~,nY] = size(Y);
C = zeros(nX,nY);
exp_a = exp(-a./eta);
exp_b = exp(-b./eta);
Array_sk_info_nfe = zeros(2*maxiter,3);
Array_iter_info   = zeros(maxiter+1,3);
ones_nX = ones(nX,1);
ones_nY = ones(nY,1);
max_subiter_sk = 1000;
Kexp_C_exp_b = zeros(nY,1);
Kexp_C = zeros(nX,nY);
logr = log(r);  logc = log(c); % this is for sk_type = 'log'
n_log = 0;
n_exp = 0;
Pi = zeros(nX,nY);
%% iter 0
iter = 0;
if K == ones(nX,nY)
    K_tag = 'ones';
else
    K_tag = 'non-ones';
    eta_logK = eta*log(K);
    logK = log(K);
end
eta_logK = eta*log(K);

%% iter 0
if ~isfield(opts,'M')
    opts.M = 8;
end
M = opts.M;
scale = gtol_ab/gtol_U;
rho_log = opts.rho_log;
rho_exp = opts.rho_exp;
%% iter 0, to compute the inexact gradient/function information
nrm_gradU = inf;
if ~isfield(opts,'tol_sub_type')
    opts.tol_sub_type = 'U_ab';
end
tol_sub_type = opts.tol_sub_type;

if ~isfield(opts,'sk_type')
    opts.sk_type = 'exp';
end
sk_type  = opts.sk_type; % exp or log
tol_sub_sk  = 1; %max(rho*scale*nrm_gradU,gtol_ab);
if rho < inf
    [f,gradU,feasi_ab,iter_sk_sub] = compute_fge_exp(U,tol_sub_sk);
else
    [f,gradU,feasi_ab,iter_sk_sub] = compute_fge_exp_inf(U,tol_sub_sk);
end
tol_sub_sk_array = zeros(maxiter+1,1);
tol_sub_sk_array(1,1) = 2;
nrm_gradU = norm(gradU,'fro');
vs = 0.49*eta;
NLS_fail = 0; 

Ef = f + vs*feasi_ab^2;
fr = Ef; Q  = 1; gamma = 0.85; 0.95; %initial setting for nonmonotone linesearch
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
% rr = gradU.*gradU;
% rr = 1;
d =  - gradU; %./sqrt(rr + 1e-12);
%% starting iterations
if rho == 12345
    dynamic_rho = 1;
    rho = 1;
else
    dynamic_rho = 0;
end
level = 0.05;
sty_neg = 0;
diff_f = inf;
for iter = 1:maxiter
    fp = f;
    tol_sub_sk_array(iter + 1,1) = feasi_ab; %
    %     if iter >= 10
    if dynamic_rho
        rho = min(1000,1.05*rho);
    end
    if feasi_ab <= gtol_ab && nrm_gradU <= gtol_U || NLS_fail == 10 ...|| isnan(nrm_gradU) 
            %|| diff_f <= 1e-16
        break;
    end
    %% store the information of latest iteration
    Up = U; %gradp = grad; fp = f;
    nrm_gradp = nrm_gradU;
    %% linesearch based on the inexact information
    descent = delta*nrm_gradp^2;
    nls = 0;
    dp = d;
    switch tol_sub_type
        case 'ab'
            %             tol_sub_sk  = 2;
            temp = max(tol_sub_sk_array(max(iter + 1 - M,1):iter+1,1)); %测试一下等于3的情况吗？
            tol_sub_sk = max([rho*temp,gtol_ab]);  %rho*nrm_gradU
        case 'U_ab'
            tol_sub_sk = max(min(rho*scale*nrm_gradU),gtol_ab); %,0.1/iter
    end
    %     tol_sub_sk = 1/iter^2;
    while 1
        U = retr(Up,dp,tau,'polar');
        %         U = U*R;
        if rho < inf
            [f,gradU,feasi_ab,iter_sk_sub] = compute_fge_exp(U,tol_sub_sk);
        else
            [f,gradU,feasi_ab,iter_sk_sub] = compute_fge_exp_inf(U,tol_sub_sk);
        end
        Ef = f + vs*feasi_ab^2;
        %         fr = inf;
        if Ef <= fr - descent - (0.5*eta - vs)*feasi_ab^2  || nls >= 5 % +  0/iter^2
            break;
        end
        nls = nls + 1;
        tau = 0.2*tau;
    end
    if nls >= 5
        NLS_fail = NLS_fail + 1;
    else
        NLs_fail = 0;
    end
    diff_f = abs(f - fp)/abs(f);
    %   nrm_gradU = norm(grad,'fro');
    if record >= 1
        str_print = strcat('iter: %2d, nrmgU: %2.1e, nrmgab: %2.1e,',...
            'iter_sk: %4d, step: %2.1e, nls: %4d, f: %14.12e \n');
        fprintf(1,str_print,...
            iter, nrm_gradU, feasi_ab, iter_sk_sub, tau, nls,f);
    end
    Array_iter_info(iter+1,:) = [nrm_gradU,feasi_ab,iter_sk_sub];
    %    rr = rr + grad.*grad;
    %     rr = 1;
    d = -gradU; %./sqrt(rr + 1e-12);
    dgrad = d - dp; % grad - gradp;
    dU    = U - Up;
    %% compute the BB stepsize
    switch BBtype
        case 'LBB'
            tau = (dU(:)'*dU(:)) / abs(dU(:)'*dgrad(:));
            %                         tau_s =  abs(dU(:)'*dgrad(:)) / (dgrad(:)'*dgrad(:));
            %                         ratio = tau/tau_s
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
                bb_new = 2/(phi23 + sqrt(phi23^2 - 4*phi13));
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
% sty_neg,iter
Array_iter_info = Array_iter_info(1:iter+1,:);
%% do postprocessing
if strcmp(sk_type,'exp')
    Pi = (exp_a.*Kexp_C).*exp_b';
end
Pi_round = Round_Pi(Pi,r,c);
%% outputs
C = compute_C(U,X,Y);
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
out.CPL = sum(sum(Pi_round.*Z));  %norm(Pi.*Z,'inf');
out.nrm_gradU = norm(grad,'fro');
out.feasi_ab  = feasi_ab;
out.iter = iter;
out.nfe = nfe;
out.total_sk_iter = total_sk_iter;
out.total_U_iter = total_U_iter;
out.Array_iter_info = Array_iter_info;
out.Array_sk_info_nfe = Array_sk_info_nfe(1:nfe,:);
out.n_log = n_log;
out.n_exp = n_exp;
out.NLS_fail = NLS_fail; 
if iter == maxiter
    stop_message = 'out of maxiter';
else
    stop_message = 'successful: kkt';
end
out.stop_message = stop_message;
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
        % f(U) = min_{a,b} (r'*a + c'*b + eta* sum_{ij} K_{ij} exp(-(a_i + b_j + ||U'(x_i - y_j)||^2)/eta))
        UtY = U'*Y;
        temp2 = sum(UtY.^2,1);
        XtU = X'*U; YtU = UtY';
        temp1 = sum(XtU.^2,2);
        C = (-2*XtU)*UtY + temp1 + temp2; % C_ij = ||U'(x_i - y_j)||^2
        C = max(C,0);
        if strcmp(K_tag,'ones')
            C_var = max(C(:)) - min(C(:));
        else
            C_eta_logK = C - eta_logK;
            C_var = max(C_eta_logK(:)) - min(C_eta_logK(:));
        end
        C_ab = max(norm(a,'inf'),norm(b,'inf'));
        if C_ab/eta >= 500 || C_var/eta >= 900
            sk_type = 'log';
        else
            sk_type = 'exp';
        end
        switch sk_type
            case 'exp'
                if iter <= 0
                    max_subiter_sk = 100;
                else
                    max_subiter_sk = 1000; 
                end
                tol_sub_sk = min(1,max(rho_exp*scale*nrm_gradU,gtol_ab)); % 之前用的0.4
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
                        tag_nan = 1;
                        break;
                    end
                    if feasi_ab <= tol_sub_sk
                        break;
                    end
                end
                n_exp = n_exp + iter_sk_sub;
                a = -eta*log(exp_a); b = -eta*log(exp_b);
                G = X*(Pi2.*XtU - exp_a.*(Kexp_C*(exp_b.*YtU))) + Y*(c.*YtU - exp_b.*(Kexp_C'*(exp_a.*XtU)));
                ab = proj_affine([a;b]);
                a = ab(1:nX); b = ab(nX+1:nX+nY);
            case 'log'
                if iter <= 0
                    max_subiter_sk = 100;
                else
                    max_subiter_sk = 100;2000;150; %opts.max_subiter_sk; 20; inf的时候不应该设置
                end
                tol_sub_sk = min(1,max(rho_log*scale*nrm_gradU,gtol_ab));
                u = a./eta; v = b./eta;
                if strcmp(K_tag,'ones')
                    C_temp = C./eta + u + v';
                else
                    C_temp = C./eta - logK + u + v';
                end
                for iter_sk_sub = 1:max_subiter_sk
                    logsum_1 = logsumexp_row(C_temp);
                    Pi2 = exp(logsum_1);
                    feasi_ab = norm(Pi2 - r,1);
                    if feasi_ab <= tol_sub_sk && iter_sk_sub >= 2
                        break;
                    end
                    du = - logr  + logsum_1;
                    u = u + du; % update u
                    C_temp = C_temp + du;
                    logsum_2 = logsumexp_column(C_temp); % main cost
                    dv = - logc  + logsum_2;
                    v = v + dv;  % update v
                    C_temp = C_temp + dv';
                    %                     sum(sum(Pi))
                end
                n_log = n_log + iter_sk_sub; 
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
        if strcmp(K_tag,'ones')
            C_var = max(C(:)) - min(C(:));
        else
            C_eta_logK = C - eta_logK;
            C_var = max(C_eta_logK(:)) - min(C_eta_logK(:));
        end
        C_ab = max(norm(a,'inf'),norm(b,'inf'));
        if C_ab/eta >= 500 || C_var/eta >= 900
            sk_type = 'log';
            n_log = n_log + 1;
        else
            sk_type = 'exp';
            n_exp = n_exp + 1;
        end
        switch sk_type
            case 'exp'
                if strcmp(K_tag,'ones')
                    Kexp_C = exp(-C./eta);
                else
                    Kexp_C = K.*exp(-C./eta);
                end
                exp_b = exp(-b./eta); %可以简化
                exp_a = r./(Kexp_C*exp_b);
                exp_b = c./(Kexp_C'*exp_a);
                Pi2 = exp_a.*(Kexp_C*exp_b);
                feasi_ab = norm(Pi2 - r,1);% max(norm(K2 - r), norm(K1 - c));
                Pi1 = c; %Pi'*ones_nX; % This is faster
                G = X*(Pi2.*XtU - exp_a.*(Kexp_C*(exp_b.*YtU))) + Y*(Pi1.*YtU - exp_b.*(Kexp_C'*(exp_a.*XtU)));
                a = -eta*log(exp_a); b = -eta*log(exp_b);
            case 'log'
                u = a./eta; v = b./eta;
                if strcmp(K_tag,'ones')
                    C_temp = C./eta + u + v';
                else
                    C_temp = C./eta - logK+ u + v';
                    %                     Kexp_C = K.*exp(-C./eta);
                end
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
        % projection onto [r' -c']x = 0 such that r'*a = c'*b;
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